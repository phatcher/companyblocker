"""Measure the recall/precision delta from the asymmetric blocking representation.

The representation under measurement is source=`short_name` /
target=`name`, against today's unconditional `name`-vs-`name` default.

Runs the actual `src/blocking` workflow (`execute_blocking_run`) twice
against a real, bounded GLEIF-source / GB-target corpus built from real
`data/gleif/matched/` and `data/gb/cleansed/` rows -- once with
`BlockingStrategyConfig.name_source="name"` (today's unconditional
`name`-vs-`name` default) and once with `name_source="short_name"` (the
asymmetric representation: source reads `company_cleanse`'s existing
`short_name` stem, target stays on its unchanged `name` column) -- then
reports the recall/precision delta between the two against the same real
`match_uri` ground truth.

This mirrors `scripts/measure_name_variant_recall.py`'s established
real-data sampling methodology directly: every real matched source row from
a random sample (fixed seed, reproducible) plus its real true target row,
plus a large random sample of real *distractor* target rows drawn from the
same real GB dataset, to keep top-k competition realistic. No name or match
label in the resulting corpus is synthetic -- only the population size is
bounded (full GB target is ~5.7M rows; a real two-run brute-force TF-IDF
comparison at that scale is impractically slow for a one-off measurement
script).

Also reports the real `short_name` null rate in the sampled source
population -- `name_source="short_name"` deliberately does not coalesce a
null `short_name` back to `name`/`name_cleansed`, keeping the representation
a single column choice rather than an operator chain with a fallback, so
those rows contribute zero candidates under that setting and
this script surfaces that rate so it isn't a silent confound in the result.

The sampled corpus is built in a scratch directory under the temp root and
removed afterwards; the report is written where `blocking.run_layout` names it,
keyed by the sampling and scoring settings.

Usage:
    .venv/Scripts/python.exe scripts/measure_short_name_recall.py \
        --source-sample 5000 --distractor-sample 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    OUTPUT_EXTEND,
    PlannedOutput,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)
from measure_name_variant_recall import add_measurement_args, measurement_settings

from blocking.contracts import BlockingRunConfig, BlockingStrategyConfig
from blocking.loader import load_dataset_descriptor
from blocking.name_transform import DEFAULT_CLEANSE_PROFILE, derive_name_forms
from blocking.run_layout import resolve_measurement_report_path
from blocking.workflow import execute_blocking_run
from validation.contracts import POPULATION_UNIVERSE
from validation.runner import pair_truth_eval_row
from workspace.data_layout import system_layer_dir
from workspace.layer_layout import layer_partition_dir, resolve_partition_dir
from workspace.roots import WorkspaceRoots, scratch_workspace

_MEASUREMENT = "short_name_recall"
_JURISDICTION = "gb"
_SOURCE_SYSTEM = "gleif"
_TARGET_SYSTEM = "gb"


def _write_partition(
    roots: WorkspaceRoots, *, system: str, layer: str, country: str, frame: pl.DataFrame
) -> Path:
    layer_dir = system_layer_dir(roots, system, layer=layer)
    partition_dir = layer_partition_dir(layer_dir, value=country)
    partition_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(partition_dir / "part-00001.parquet")
    return layer_dir


def _write_match_metadata(
    layer_dir: Path, *, source_system: str, target_systems: list[str]
) -> None:
    metadata = {"source_system": source_system, "target_systems": target_systems}
    (layer_dir / "_match_metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )


def build_sample_corpus(
    *, roots: WorkspaceRoots, source_sample: int, distractor_sample: int, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (sampled_source, target_frame), all real rows."""

    matched_layer_dir = system_layer_dir(roots, _SOURCE_SYSTEM, layer="matched")
    matched_dir = resolve_partition_dir(
        matched_layer_dir, value=_JURISDICTION
    ) or layer_partition_dir(matched_layer_dir, value=_JURISDICTION)
    matched_files = sorted(matched_dir.glob("*.parquet"))
    if not matched_files:
        raise FileNotFoundError(f"No matched parquet files at {matched_dir}")

    matched = (
        pl.scan_parquet([str(f) for f in matched_files])
        .filter(pl.col("match_uri").is_not_null())
        .select(["system_uri", "name", "jurisdiction_code", "match_uri"])
        .collect()
    )
    # `matched/` carries no cleansed name, so the sample's `short_name` is
    # derived here from the raw name, the way the run itself derives it.
    sampled_source = derive_name_forms(
        matched.sample(n=min(source_sample, matched.height), seed=seed, shuffle=True),
        profile=DEFAULT_CLEANSE_PROFILE,
    ).select(["system_uri", "name", "short_name", "jurisdiction_code", "match_uri"])
    true_target_ids = set(sampled_source.get_column("match_uri").to_list())

    target_layer_dir = system_layer_dir(roots, _TARGET_SYSTEM, layer="cleansed")
    target_dir = resolve_partition_dir(
        target_layer_dir, value=_JURISDICTION
    ) or layer_partition_dir(target_layer_dir, value=_JURISDICTION)
    target_files = sorted(target_dir.glob("*.parquet"))
    if not target_files:
        raise FileNotFoundError(f"No cleansed parquet files at {target_dir}")

    target_all = (
        pl.scan_parquet([str(f) for f in target_files])
        .select(["system_uri", "name", "jurisdiction_code"])
        .collect()
    )
    true_targets = target_all.filter(pl.col("system_uri").is_in(list(true_target_ids)))
    # sampled_source rows whose true target didn't resolve in the target
    # dataset (should be rare/nonexistent, but keep the corpus internally
    # consistent -- a source row referencing a target we didn't load would
    # register as a guaranteed miss for both runs, diluting the comparison
    # with a data-availability artifact rather than a real recall difference).
    resolved_target_ids = set(true_targets.get_column("system_uri").to_list())
    sampled_source = sampled_source.filter(
        pl.col("match_uri").is_in(list(resolved_target_ids))
    )

    distractor_pool = target_all.filter(
        ~pl.col("system_uri").is_in(list(true_target_ids))
    )
    distractors = distractor_pool.sample(
        n=min(distractor_sample, distractor_pool.height), seed=seed, shuffle=True
    )

    target_frame = pl.concat([true_targets, distractors], how="vertical_relaxed")

    return sampled_source, target_frame


def write_corpus(
    tmp_roots: WorkspaceRoots,
    *,
    sampled_source: pl.DataFrame,
    target_frame: pl.DataFrame,
) -> Path:
    matched_dir = _write_partition(
        tmp_roots,
        system=_SOURCE_SYSTEM,
        layer="matched",
        country=_JURISDICTION,
        frame=sampled_source,
    )
    _write_match_metadata(
        matched_dir, source_system=_SOURCE_SYSTEM, target_systems=[_TARGET_SYSTEM]
    )
    _write_partition(
        tmp_roots,
        system=_TARGET_SYSTEM,
        layer="cleansed",
        country=_JURISDICTION,
        frame=target_frame,
    )
    return tmp_roots.data


def run_comparison(
    tmp_roots: WorkspaceRoots, *, top_k: int, min_similarity: float
) -> tuple[dict[str, object], dict[str, object]]:
    source = load_dataset_descriptor(roots=tmp_roots, system=_SOURCE_SYSTEM)
    target = load_dataset_descriptor(
        roots=tmp_roots, system=_TARGET_SYSTEM, require_ground_truth=False
    )

    def _run(name_source: str) -> dict[str, object]:
        strategy = BlockingStrategyConfig(
            representation="tfidf",
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=None,
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
            name_source=name_source,
        )
        config = BlockingRunConfig(
            roots=tmp_roots,
            prepared_base_dir=None,
            source=source,
            target=target,
            countries=None,
            strategy=strategy,
            # Nothing is written: this reads the in-memory result, so the run
            # needs no destination and resolves none.
        )
        result = execute_blocking_run(config)
        assert result.pair_truth_eval is not None  # nosec B101 - narrows Optional for mypy
        row = pair_truth_eval_row(result.pair_truth_eval, POPULATION_UNIVERSE)
        return {
            "tp": row["tp"],
            "fp": row["fp"],
            "fn": row["fn"],
            "precision": row["precision"],
            "recall": row["recall"],
            "truth_pairs": row["truth_pairs"],
            "predicted_pairs": row["predicted_pairs"],
        }

    return _run("name"), _run("short_name")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_measurement_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    settings = measurement_settings(args)
    report_path = resolve_measurement_report_path(
        roots, measurement=_MEASUREMENT, settings=settings
    )

    if args.dry_run:
        report_dry_run("measure_short_name_recall", data_dir=roots.data, **settings)
        report_output_plan(
            "[dry-run]  ",
            [
                PlannedOutput(
                    roots.temp,
                    OUTPUT_EXTEND,
                    note="sampled corpus built in a new directory here, removed afterwards",
                ),
                PlannedOutput(report_path, OUTPUT_CLEAR, note="measurement report"),
            ],
        )
        return 0

    sampled_source, target_frame = build_sample_corpus(
        roots=roots,
        source_sample=args.source_sample,
        distractor_sample=args.distractor_sample,
        seed=args.seed,
    )
    null_or_blank_short_name = (
        sampled_source.get_column("short_name")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .eq("")
        .sum()
    )
    print(
        f"corpus: {sampled_source.height} matched source rows, "
        f"{target_frame.height} target rows "
        f"({sampled_source.get_column('match_uri').n_unique()} true targets + "
        f"distractors); {null_or_blank_short_name} of {sampled_source.height} "
        f"({null_or_blank_short_name / max(sampled_source.height, 1):.2%}) sampled "
        "source rows have a null/blank short_name (no fallback -- these "
        "contribute zero candidates under name_source='short_name')"
    )

    with scratch_workspace(roots, prefix=_MEASUREMENT) as tmp_roots:
        write_corpus(
            tmp_roots, sampled_source=sampled_source, target_frame=target_frame
        )
        baseline, asymmetric = run_comparison(
            tmp_roots, top_k=args.top_k, min_similarity=args.min_similarity
        )

    print("-" * 70)
    print(f"{'metric':<16}{'name/name':>16}{'short_name/name':>20}{'delta':>12}")
    for key in ("tp", "fp", "fn", "recall", "precision", "predicted_pairs"):
        b = cast(float, baseline[key])
        v = cast(float, asymmetric[key])
        delta = v - b
        if isinstance(b, float):
            print(f"{key:<16}{b:>16.4f}{v:>20.4f}{delta:>12.4f}")
        else:
            print(f"{key:<16}{b:>16}{v:>20}{delta:>12}")

    payload = {
        "settings": settings,
        "baseline_name_name": baseline,
        "asymmetric_short_name_name": asymmetric,
        "null_or_blank_short_name_count": int(null_or_blank_short_name),
        "sampled_source_count": sampled_source.height,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote full measurement to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
