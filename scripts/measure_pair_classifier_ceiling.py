"""Measure the pair classifier's ceiling on real pairs.

Runs `company_classify`'s dedicated pair classifier (`TfidfPairMlpClassifier`
via `cross_train_pairs`) and its single-text baseline adapter
(`TfidfLogRegClassifier`/`TfidfMlpClassifier` via `cross_train_pair_baseline`)
over real observed name-variant pairs from one cleansed system
(`RealAliasPairProducer` over Wikidata's canonical-stage names sidecar), on a
fixed, recorded entity-level split (`EntitySplitter`), then records both
sides' `MetricBundle`s side by side so the classifier's ceiling over the
baseline is a number rather than a phrase.

Why Wikidata/GB: `RealAliasPairProducer`'s contract groups an entity's variants by the
identity `load_real_alias_variants()` attaches to each row, not by the sidecar's raw
`system_uri` column -- that column is the row's own derived URI, one per name variant
(`name://<system>/<id>/<hash>`), so grouping on it directly would put every variant in its
own singleton group. `load_real_alias_variants()` resolves each row's `system_uri` back to
its entity via `workspace.match_resolution.entity_of` first, and Wikidata's
canonical-stage names sidecar (`data/wikidata/canonical/*/names/wikidata-names-*.parquet`)
resolves cleanly: an entity with several recorded `official`/`short`/`alias`/`label` forms
carries one row per form, all decomposing to the same entity URI (GB alone: 10,603 distinct
entities, 4,754 of them with more than one recorded form). GLEIF's equivalent sidecar does
not -- its `system_uri` column is a per-name-row synthetic id in a scheme `entity_of` cannot
decompose at all, and the entity id instead lives in its `source_uri`/`LEI` columns, a
genuine schema mismatch with what `pairs.py` documents the sidecar shape to be -- so GLEIF was tried first and dropped for this reason, not
for size. No name or label in the run is synthetic; only the population is bounded to one
jurisdiction, so this stays a one-off measurement rather than an hours-long job.

This never writes under `data/`, and never re-runs acquire/shard.

Usage:
    .venv/Scripts/python.exe scripts/measure_pair_classifier_ceiling.py \
        --json-out tmp/pair_classifier_ceiling/wikidata_gb_ceiling.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    RoleSetting,
    add_declared_arguments,
    add_dry_run_arg,
    add_role_settings,
    add_workspace_roots_args,
    declared_settings,
    report_output_plan,
    report_resolved_settings,
    resolve_declared_settings,
    resolve_role_settings,
    resolve_workspace_roots_from_args,
    resolved_setting_values,
    run_reporting_argument_errors,
)
from company_classify import (
    EntitySplitter,
    NameVariant,
    NegativeSamplingConfig,
    NegativeStrategy,
    RealAliasPairProducer,
    SplitRatios,
    TfidfLogRegClassifier,
    TfidfMlpClassifier,
    TfidfPairMlpClassifier,
    cross_train_pair_baseline,
    cross_train_pairs,
    name_variants_from_rows,
    split_pairs,
    validate_pairs,
)
from company_classify._cli_helper import SETTINGS as CLASSIFY_SETTINGS

from workspace.data_layout import canonical_snapshot_dir
from workspace.layer_layout import resolve_name_files
from workspace.match_resolution import entity_of
from workspace.roots import WorkspaceRoots

_SYSTEM = "wikidata"
_CANONICAL_DATE = "2026-07-16"
_JURISDICTION = "GB"

DECLARATIONS = declared_settings(CLASSIFY_SETTINGS)
"""Every setting this script and its sibling encoder script may take."""

SETTING_SURFACE: dict[str, dict[str, object]] = {
    "split_seed": {"flag": "--split-seed"},
    "split_train_ratio": {"flag": "--split-train-ratio"},
    "split_validation_ratio": {"flag": "--split-validation-ratio"},
    "negative_strategy": {"flag": "--negative-strategy"},
    "negatives_per_positive": {"flag": "--negatives-per-positive"},
    "negative_seed": {"flag": "--negative-seed"},
    "hard_pool_size": {"flag": "--hard-pool-size"},
}
"""Where this script (and `measure_pooled_subword_encoder_ceiling.py`, which
imports it) takes each declared setting: every one a flag, no default of its
own -- the package's own default is the one reported."""


def resolve_splitter_and_negatives(
    values: dict[str, Any],
) -> tuple[EntitySplitter, NegativeSamplingConfig]:
    """The `EntitySplitter`/`NegativeSamplingConfig` `values` (from
    `resolved_setting_values`) describe, shared by both ceiling scripts so
    the same resolved settings build the same objects either way."""
    splitter = EntitySplitter(
        seed=int(values["split_seed"]),
        ratios=SplitRatios(
            train=float(values["split_train_ratio"]),
            validation=float(values["split_validation_ratio"]),
        ),
    )
    negatives = NegativeSamplingConfig(
        strategy=NegativeStrategy(str(values["negative_strategy"])),
        negatives_per_positive=float(values["negatives_per_positive"]),
        hard_pool_size=int(values["hard_pool_size"]),
        seed=int(values["negative_seed"]),
    )
    return splitter, negatives


def resolve_relative_to_checkout(
    roots: WorkspaceRoots, value: str | None
) -> Path | None:
    """`value` resolved against `roots.checkout`, never the working directory;
    `None` when no path was given."""
    if value is None:
        return None
    return (roots.checkout / Path(value)).resolve()


def load_real_alias_variants(
    roots: WorkspaceRoots, *, system: str, canonical_date: str, jurisdiction_code: str
) -> list[NameVariant]:
    """Load one jurisdiction's real observed name-variant rows for `system`.

    Reads `data/<system>/canonical/<canonical_date>/names/<system>-names-*.parquet`
    (the canonical-stage sidecar `RealAliasPairProducer` is built to consume), filtered
    to `jurisdiction_code`, resolves each row's own `system_uri` back to its entity via
    `workspace.match_resolution.entity_of` (a name row in one hop, a perturbed row in
    two, an entity row in none), and adapts the rows via `name_variants_from_rows()`.
    Two variants of one entity land under the same resolved identity this way; passing the
    row's raw `system_uri` through unresolved would not, since that URI is the row's own.
    A URI `entity_of` cannot decompose (an old-form `name://<hash>` sidecar, or a row from
    a system that plays no part in `workspace`'s derivation scheme) raises rather than
    being read as an entity.
    """
    canonical_dir = canonical_snapshot_dir(
        roots, system=system, run_date=canonical_date
    )
    files = resolve_name_files(canonical_dir, system_code=system)
    if not files:
        raise FileNotFoundError(f"No names sidecar parquet under {canonical_dir}")

    frame = (
        pl.scan_parquet([str(path) for path in files])
        .filter(pl.col("jurisdiction_code") == jurisdiction_code)
        .select(["system_uri", "name", "name_type"])
        .collect()
    )
    rows = (
        {**row, "system_uri": entity_of(str(row["system_uri"])).uri}
        for row in frame.iter_rows(named=True)
    )
    return name_variants_from_rows(rows)


def build_ceiling_report(
    *,
    system: str,
    jurisdiction_code: str,
    canonical_date: str,
    splitter: EntitySplitter,
    negatives: NegativeSamplingConfig,
    variants: list[NameVariant],
) -> dict[str, Any]:
    """Run the pair classifier and its baseline over `variants`, same pairs and split.

    Raises `PairContractError` (via `validate_pairs()`) if the generated pairs break
    the split-leakage, structural, or ambiguity contract `pairs.py` defines.
    """
    producer = RealAliasPairProducer(
        variants=variants, splitter=splitter, negatives=negatives
    )
    pairs = producer.generate()
    validate_pairs(pairs, splitter=splitter)

    split = split_pairs(pairs)

    pair_start = time.perf_counter()
    pair_results = cross_train_pairs(split, {"pair_mlp": TfidfPairMlpClassifier()})
    pair_elapsed = time.perf_counter() - pair_start

    baseline_start = time.perf_counter()
    baseline_results = cross_train_pair_baseline(
        split,
        {
            "tfidf_logreg": TfidfLogRegClassifier(),
            "tfidf_mlp": TfidfMlpClassifier(),
        },
    )
    baseline_elapsed = time.perf_counter() - baseline_start

    pair_metrics = pair_results["pair_mlp"].test_metrics
    best_baseline_name = max(
        baseline_results, key=lambda name: baseline_results[name].test_metrics.f1
    )
    best_baseline_metrics = baseline_results[best_baseline_name].test_metrics

    return {
        "run": {
            "system": system,
            "jurisdiction_code": jurisdiction_code,
            "canonical_date": canonical_date,
            "producer": "RealAliasPairProducer",
            "split_seed": splitter.seed,
            "split_ratios": {
                "train": splitter.ratios.train,
                "validation": splitter.ratios.validation,
                "test": splitter.ratios.test,
            },
            "negative_strategy": negatives.strategy.value,
            "negatives_per_positive": negatives.negatives_per_positive,
            "negative_seed": negatives.seed,
            "variant_rows": len(variants),
            "pair_counts": {
                "total": len(pairs),
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
            "fit_seconds": {
                "pair_classifier": pair_elapsed,
                "baseline": baseline_elapsed,
            },
        },
        "pair_classifier": {
            "name": "pair_mlp",
            "validation_metrics": asdict(pair_results["pair_mlp"].validation_metrics),
            "test_metrics": asdict(pair_metrics),
        },
        "baseline": {
            name: {
                "validation_metrics": asdict(result.validation_metrics),
                "test_metrics": asdict(result.test_metrics),
            }
            for name, result in baseline_results.items()
        },
        "ceiling": {
            "best_baseline": best_baseline_name,
            "classifier_f1": pair_metrics.f1,
            "baseline_f1": best_baseline_metrics.f1,
            "f1_delta": pair_metrics.f1 - best_baseline_metrics.f1,
            "classifier_pr_auc": pair_metrics.pr_auc,
            "baseline_pr_auc": best_baseline_metrics.pr_auc,
            "pr_auc_delta": pair_metrics.pr_auc - best_baseline_metrics.pr_auc,
            "classifier_beats_baseline": pair_metrics.f1 > best_baseline_metrics.f1,
        },
    }


_DATE_SETTINGS = (
    RoleSetting(
        "date",
        "source",
        f"The canonical snapshot read (default: {_CANONICAL_DATE}).",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_roots_args(parser)
    parser.add_argument("--system", default=_SYSTEM)
    add_role_settings(parser, _DATE_SETTINGS)
    parser.add_argument("--jurisdiction", default=_JURISDICTION)
    add_declared_arguments(parser, DECLARATIONS, SETTING_SURFACE)
    parser.add_argument(
        "--json-out",
        default=None,
        help="Where to write the full measurement report. A relative path resolves "
        "against the checkout.",
    )
    add_dry_run_arg(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.canonical_date = (
        resolve_role_settings(args, _DATE_SETTINGS)["source_date"] or _CANONICAL_DATE
    )
    roots = resolve_workspace_roots_from_args(args)

    resolved = resolve_declared_settings(args, DECLARATIONS, SETTING_SURFACE)
    json_out = resolve_relative_to_checkout(roots, args.json_out)
    input_dir = canonical_snapshot_dir(
        roots, system=args.system, run_date=args.canonical_date
    )

    if args.dry_run:
        report_resolved_settings(
            "measure_pair_classifier_ceiling",
            resolved,
            system=args.system,
            jurisdiction=args.jurisdiction,
            canonical_date=args.canonical_date,
            input_dir=input_dir,
            json_out=json_out,
        )
        if json_out is not None:
            report_output_plan(
                "[dry-run]  ",
                [PlannedOutput(json_out, OUTPUT_CLEAR, note="full measurement report")],
            )
        else:
            print("[dry-run]   --json-out not given; nothing would be written")
        return 0

    variants = load_real_alias_variants(
        roots,
        system=args.system,
        canonical_date=args.canonical_date,
        jurisdiction_code=args.jurisdiction,
    )
    print(
        f"{len(variants)} real name-variant rows loaded for "
        f"{args.system}/{args.jurisdiction} ({args.canonical_date})"
    )

    splitter, negatives = resolve_splitter_and_negatives(
        resolved_setting_values(resolved)
    )

    report = build_ceiling_report(
        system=args.system,
        jurisdiction_code=args.jurisdiction,
        canonical_date=args.canonical_date,
        splitter=splitter,
        negatives=negatives,
        variants=variants,
    )

    print(json.dumps(report["run"], indent=2))
    print(json.dumps(report["ceiling"], indent=2))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote full measurement to {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
