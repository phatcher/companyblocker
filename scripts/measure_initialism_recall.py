"""Measure how often real matched pairs are initialism-shaped.

Read-only analysis script, not a build task. Measures, against a real
`matched/` ground-truth layer (produced by `src/acquisition/match_ops.py`),
how often a matched source/target pair has *zero token overlap* between
their cleansed names but one side's name reduces to the per-token initials
of the other side (e.g. "ibm" vs "international business machines"). That
rate is the evidence for whether an initialism-based candidate-generation
signal is worth building: no such signal exists today, and this script does
not build, index, or wire any candidate-generation code; it only measures.

Usage:
    .venv/Scripts/python.exe scripts/measure_initialism_recall.py \
        --source gleif

Reuses the same short_name/company_type convention
`packages/company_cleanse/src/company_cleanse/extract.py`'s
`_initials_excluding_company_type`/`_derive_acronym` already established
in-record: `short_name` is the cleansed name with its legal-suffix company
type already stripped, so tokenizing it (plain whitespace split, matching
that function's own `companyname.strip().split()`) and taking each token's
first letter reproduces the same initials computation, just applied
across two different records instead of within one.
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
    add_dry_run_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.derived_uri import parse_uri
from workspace.layer_layout import (
    layer_partition_dir,
    partition_values,
    resolve_partition_dir,
)
from workspace.reference import InvalidReferenceError
from workspace.roots import WorkspaceRoots


def _tokenize(name: str | None) -> tuple[str, ...]:
    if not name:
        return ()
    return tuple(token for token in name.strip().lower().split() if token)


def _is_initialism_shaped(
    source_tokens: tuple[str, ...], target_tokens: tuple[str, ...]
) -> bool:
    """True iff zero token overlap and one side reduces to the other's initials.

    "Reduces to the initials of" is checked both directions: does the
    source name (tokens concatenated, no spaces) equal the initials
    computed from the target's own tokens, or vice versa.
    """
    if not source_tokens or not target_tokens:
        return False
    if set(source_tokens) & set(target_tokens):
        return False
    source_initials = "".join(token[0] for token in source_tokens)
    target_initials = "".join(token[0] for token in target_tokens)
    source_joined = "".join(source_tokens)
    target_joined = "".join(target_tokens)
    return source_initials == target_joined or target_initials == source_joined


def _discover_jurisdictions(matched_dir: Path) -> list[str]:
    return partition_values(matched_dir)


def _resolve_target_system(matched_partition: Path) -> str | None:
    """Infer the target system name from the scheme of a non-null match_uri.

    `match_ops.py` writes `match_uri` values as `<target_system>://<id>`
    (the target's own `system_uri`), so the URI scheme is the target
    system's directory name under `data/`.
    """
    files = sorted(matched_partition.glob("*.parquet"))
    if not files:
        return None
    lf = pl.scan_parquet([str(f) for f in files]).select("match_uri")
    sample = lf.filter(pl.col("match_uri").is_not_null()).limit(1).collect()
    if sample.is_empty():
        return None
    try:
        return parse_uri(sample["match_uri"][0]).system
    except InvalidReferenceError:
        return None


@dataclass
class JurisdictionMeasurement:
    jurisdiction: str
    target_system: str | None
    matched_pairs: int
    pairs_with_both_names: int
    zero_overlap_pairs: int
    initialism_shaped_pairs: int
    sample_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def initialism_share_of_matched(self) -> float:
        if self.pairs_with_both_names == 0:
            return 0.0
        return self.initialism_shaped_pairs / self.pairs_with_both_names

    @property
    def initialism_share_of_zero_overlap(self) -> float:
        if self.zero_overlap_pairs == 0:
            return 0.0
        return self.initialism_shaped_pairs / self.zero_overlap_pairs

    def as_dict(self) -> dict[str, object]:
        return {
            "jurisdiction": self.jurisdiction,
            "target_system": self.target_system,
            "matched_pairs": self.matched_pairs,
            "pairs_with_both_names": self.pairs_with_both_names,
            "zero_overlap_pairs": self.zero_overlap_pairs,
            "initialism_shaped_pairs": self.initialism_shaped_pairs,
            "initialism_share_of_matched": round(self.initialism_share_of_matched, 6),
            "initialism_share_of_zero_overlap": round(
                self.initialism_share_of_zero_overlap, 6
            ),
            "sample_pairs": self.sample_pairs,
        }


def measure_jurisdiction(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    jurisdiction: str,
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
            zero_overlap_pairs=0,
            initialism_shaped_pairs=0,
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
    zero_overlap_pairs = 0
    initialism_shaped_pairs = 0
    sample_pairs: list[tuple[str, str]] = []

    for source_name, target_name in zip(
        joined["source_name"].to_list(), joined["target_name"].to_list()
    ):
        source_tokens = _tokenize(source_name)
        target_tokens = _tokenize(target_name)
        if not source_tokens or not target_tokens:
            continue
        pairs_with_both_names += 1
        if set(source_tokens) & set(target_tokens):
            continue
        zero_overlap_pairs += 1
        if _is_initialism_shaped(source_tokens, target_tokens):
            initialism_shaped_pairs += 1
            if len(sample_pairs) < sample_size:
                sample_pairs.append((source_name, target_name))

    return JurisdictionMeasurement(
        jurisdiction=jurisdiction,
        target_system=target_system,
        matched_pairs=matched_pairs,
        pairs_with_both_names=pairs_with_both_names,
        zero_overlap_pairs=zero_overlap_pairs,
        initialism_shaped_pairs=initialism_shaped_pairs,
        sample_pairs=sample_pairs,
    )


def measure_source_system(
    *, roots: WorkspaceRoots, source_system: str, jurisdictions: list[str] | None = None
) -> list[JurisdictionMeasurement]:
    matched_dir = system_layer_dir(roots, source_system, layer=MATCHED_LAYER_NAME)
    if not matched_dir.exists():
        raise FileNotFoundError(f"No matched/ directory found at {matched_dir}.")

    if jurisdictions is None:
        jurisdictions = _discover_jurisdictions(matched_dir)

    return [
        measure_jurisdiction(
            roots=roots, source_system=source_system, jurisdiction=jurisdiction
        )
        for jurisdiction in jurisdictions
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how often real matched/ ground-truth pairs are "
            "initialism-shaped (zero token overlap, one side reduces to the "
            "other's per-token initials). Read-only; "
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
            "Report the resolved source system, jurisdictions and output "
            "path, without reading any matched/cleansed data or writing "
            "anything."
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
    json_out = _resolve_json_out(roots, args.json_out)

    jurisdictions = (
        [j.strip() for j in args.jurisdictions.split(",") if j.strip()]
        if args.jurisdictions
        else None
    )

    if args.dry_run:
        report_dry_run(
            "measure_initialism_recall",
            source_system=args.source_system,
            jurisdictions=jurisdictions or "discovered",
        )
        if json_out is not None:
            report_output_plan(
                "[dry-run] measure_initialism_recall:",
                [PlannedOutput(json_out, OUTPUT_CLEAR)],
            )
        return 0

    results = measure_source_system(
        roots=roots, source_system=args.source_system, jurisdictions=jurisdictions
    )

    total_matched = sum(r.matched_pairs for r in results)
    total_with_names = sum(r.pairs_with_both_names for r in results)
    total_zero_overlap = sum(r.zero_overlap_pairs for r in results)
    total_initialism = sum(r.initialism_shaped_pairs for r in results)

    print(f"source_system={args.source_system}")
    print(
        f"{'jurisdiction':<14}{'target':<18}{'matched':>10}{'named':>10}"
        f"{'zero_overlap':>14}{'initialism':>12}{'%_of_named':>12}"
    )
    for r in results:
        pct = r.initialism_share_of_matched * 100
        print(
            f"{r.jurisdiction:<14}{r.target_system!s:<18}{r.matched_pairs:>10}"
            f"{r.pairs_with_both_names:>10}{r.zero_overlap_pairs:>14}"
            f"{r.initialism_shaped_pairs:>12}{pct:>11.4f}%"
        )

    overall_pct = (
        (total_initialism / total_with_names * 100) if total_with_names else 0.0
    )
    zero_overlap_pct = (
        (total_zero_overlap / total_with_names * 100) if total_with_names else 0.0
    )
    initialism_of_zero_overlap_pct = (
        (total_initialism / total_zero_overlap * 100) if total_zero_overlap else 0.0
    )
    print("-" * 90)
    print(
        f"TOTAL matched={total_matched} named_pairs={total_with_names} "
        f"zero_overlap={total_zero_overlap} ({zero_overlap_pct:.4f}% of named) "
        f"initialism_shaped={total_initialism} ({overall_pct:.4f}% of named, "
        f"{initialism_of_zero_overlap_pct:.4f}% of zero-overlap pairs)"
    )

    sample_shown = 0
    for r in results:
        for source_name, target_name in r.sample_pairs:
            if sample_shown >= 15:
                break
            print(f"  sample [{r.jurisdiction}]: {source_name!r} <-> {target_name!r}")
            sample_shown += 1

    if json_out is not None:
        payload = {
            "source_system": args.source_system,
            "jurisdictions": [r.as_dict() for r in results],
            "total_matched_pairs": total_matched,
            "total_pairs_with_both_names": total_with_names,
            "total_zero_overlap_pairs": total_zero_overlap,
            "total_initialism_shaped_pairs": total_initialism,
            "initialism_share_of_named_pct": round(overall_pct, 6),
            "initialism_share_of_zero_overlap_pct": round(
                initialism_of_zero_overlap_pct, 6
            ),
        }
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote full measurement to {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
