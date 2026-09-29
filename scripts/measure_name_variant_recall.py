"""Measure real name-variant recovery as its own labelled run.

Multi-key target-index expansion folded into a shared run's own metrics is
gone: expanding the target index with recorded names made "did a previous
name reach the current company" true by construction, spending the ground
truth those rows carry instead of measuring against it. This script instead
runs the actual `src/blocking` workflow twice against a real, bounded GLEIF-source /
GB-target corpus built from real `data/gleif/matched/`, `data/gb/cleansed/`,
and `data/gb/canonical/.../gb-names-*.parquet` rows -- once as the ordinary
canonical `execute_blocking_run()` baseline (source primary records against
target primary records), and once as `execute_name_variant_recovery_run()`'s
own labelled pairing (GB's own recorded name variants scored against GB's
own primary records, ground truth taken from each variant row's own
`source_uri` back-reference) -- and reports both runs' own figures rather
than a blended delta between them, since the two answer different
questions: the baseline is cross-system match recall, the recovery run is
"does a recorded former name still resolve to the record it came from".

GLEIF/GB was chosen because it's the only pairing in this repo's real data
where the *source* has ground truth (`data/gleif/matched/_match_metadata.json`
lists `gb` as a matched target system) *and* the *target* has a
`*-names-*.parquet` sidecar (`data/gb/canonical/*/gb-names-*.parquet`, 647,753
real Companies House `PREVIOUS_NAME` rows).

Why sampled rather than the full corpus: the full GB target side is ~5.7M
rows; brute-force TF-IDF/cosine target-index construction and scoring at
that scale is a multi-x-minute-to-hour job unsuited to a one-off measurement
script, and `execute_blocking_run` is run twice here. Instead this builds a
bounded-but-real corpus: every real matched source row from a random sample
(fixed seed, so reruns are reproducible) plus its real true target row, plus
a large random sample of real *distractor* target rows drawn from the same
real GB dataset (default 50,000) to keep top-k competition realistic. No
name, previous-name, or match label in the resulting corpus is synthetic --
only the population size is bounded. This mirrors this repo's established
precedent for a read-only recall-measurement script (see
`scripts/measure_initialism_recall.py`), just needing to actually exercise
the real workflow/index-construction code path (unlike that script's pure
token-level check) since variant expansion lives in target-index
construction, not a standalone lexical rule.

The sampled corpus is built in a scratch directory under the temp root and
removed afterwards; the report is written where `blocking.run_layout` names it,
keyed by the sampling and scoring settings.

Usage:
    .venv/Scripts/python.exe scripts/measure_name_variant_recall.py \
        --source-sample 5000 --distractor-sample 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    OUTPUT_EXTEND,
    PlannedOutput,
    add_dry_run_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from blocking.contracts import BlockingRunConfig, BlockingStrategyConfig
from blocking.loader import load_dataset_descriptor
from blocking.run_layout import resolve_measurement_report_path
from blocking.workflow import execute_blocking_run, execute_name_variant_recovery_run
from validation.contracts import POPULATION_UNIVERSE
from validation.runner import pair_truth_eval_row
from workspace.data_file_naming import COMPANION_NAMES, companion_data_file_glob
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    canonical_snapshot_dir,
    system_layer_dir,
)
from workspace.layer_layout import (
    layer_partition_dir,
    resolve_name_files,
    resolve_partition_dir,
)
from workspace.roots import WorkspaceRoots, scratch_workspace

_MEASUREMENT = "name_variant_recall"
_JURISDICTION = "gb"
_SOURCE_SYSTEM = "gleif"
_TARGET_SYSTEM = "gb"
_TARGET_NAME_FILE_GLOB = companion_data_file_glob(
    system_code=_TARGET_SYSTEM, family=COMPANION_NAMES
)
_TARGET_CANONICAL_DATE = "2026-06-01"


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


def _write_names_sidecar(
    roots: WorkspaceRoots, *, system: str, frame: pl.DataFrame
) -> None:
    sidecar_dir = canonical_snapshot_dir(
        roots, system=system, run_date=_TARGET_CANONICAL_DATE
    )
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(sidecar_dir / f"{system}-names-001.parquet")


def build_sample_corpus(
    *, roots: WorkspaceRoots, source_sample: int, distractor_sample: int, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Returns (sampled_source, target_frame, variant_frame), all real rows."""

    matched_layer_dir = system_layer_dir(
        roots, _SOURCE_SYSTEM, layer=MATCHED_LAYER_NAME
    )
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
    sampled_source = matched.sample(
        n=min(source_sample, matched.height), seed=seed, shuffle=True
    )
    true_target_ids = set(sampled_source.get_column("match_uri").to_list())

    target_layer_dir = system_layer_dir(
        roots, _TARGET_SYSTEM, layer=CLEANSED_LAYER_NAME
    )
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
    # register as a guaranteed miss for both baseline and variant runs,
    # diluting the comparison with a data-availability artifact rather than
    # a real recall difference).
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

    names_dir = system_layer_dir(roots, _TARGET_SYSTEM, layer=CANONICAL_LAYER_NAME)
    names_files: list[Path] = []
    if names_dir.exists():
        for dated_dir in sorted(names_dir.iterdir()):
            if dated_dir.is_dir():
                names_files = resolve_name_files(dated_dir, system_code=_TARGET_SYSTEM)
    if not names_files:
        raise FileNotFoundError(
            f"No {_TARGET_NAME_FILE_GLOB} sidecar found under {names_dir}"
        )

    # A names sidecar row's own `system_uri` names *that row* (a per-row
    # identity, `name://<hash>`), not the entity it is a variant of --
    # joining on it against the corpus' entity ids silently returns nothing.
    # `source_uri` is the
    # back-reference to the primary entity's own `system_uri` (see
    # `company_vectorize.name_variant_index.expand_target_frame_with_name_variants`,
    # which this script's `variant_frame` output feeds straight into, and
    # already prefers it the same way); a sidecar not yet migrated to that
    # per-row shape has no `source_uri` column at all, so its own
    # `system_uri` -- still the primary's identity there -- is the fallback.
    names_frame = pl.scan_parquet([str(f) for f in names_files])
    corpus_join_col = (
        "source_uri"
        if "source_uri" in names_frame.collect_schema().names()
        else "system_uri"
    )
    target_ids_in_corpus = target_frame.select(
        pl.col("system_uri").alias(corpus_join_col)
    )
    variant_frame = names_frame.join(
        target_ids_in_corpus.lazy(), on=corpus_join_col, how="inner"
    ).collect()

    return sampled_source, target_frame, variant_frame


def write_corpus(
    tmp_roots: WorkspaceRoots,
    *,
    sampled_source: pl.DataFrame,
    target_frame: pl.DataFrame,
    variant_frame: pl.DataFrame,
) -> Path:
    matched_dir = _write_partition(
        tmp_roots,
        system=_SOURCE_SYSTEM,
        layer=MATCHED_LAYER_NAME,
        country=_JURISDICTION,
        frame=sampled_source,
    )
    _write_match_metadata(
        matched_dir, source_system=_SOURCE_SYSTEM, target_systems=[_TARGET_SYSTEM]
    )
    _write_partition(
        tmp_roots,
        system=_TARGET_SYSTEM,
        layer=CLEANSED_LAYER_NAME,
        country=_JURISDICTION,
        frame=target_frame,
    )
    _write_names_sidecar(tmp_roots, system=_TARGET_SYSTEM, frame=variant_frame)
    return tmp_roots.data


def _row_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        "tp": row["tp"],
        "fp": row["fp"],
        "fn": row["fn"],
        "precision": row["precision"],
        "recall": row["recall"],
        "truth_pairs": row["truth_pairs"],
        "predicted_pairs": row["predicted_pairs"],
    }


def run_comparison(
    tmp_roots: WorkspaceRoots, *, top_k: int, min_similarity: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Returns `(baseline_metrics, self_recovery_metrics)` -- two separate
    labelled runs, never one run's metrics folded into the other's (see this
    module's own docstring for what each measures).
    """

    source = load_dataset_descriptor(roots=tmp_roots, system=_SOURCE_SYSTEM)
    target = load_dataset_descriptor(
        roots=tmp_roots, system=_TARGET_SYSTEM, require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )

    baseline_config = BlockingRunConfig(
        roots=tmp_roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
        # Nothing is written: this reads the in-memory result, so the run needs
        # no destination and resolves none.
    )
    baseline_result = execute_blocking_run(baseline_config)
    assert baseline_result.pair_truth_eval is not None  # nosec B101 - narrows Optional for mypy
    baseline_metrics = _row_metrics(
        pair_truth_eval_row(baseline_result.pair_truth_eval, POPULATION_UNIVERSE)
    )

    recovery_result = execute_name_variant_recovery_run(
        target, strategy=strategy, roots=tmp_roots
    )
    assert recovery_result.pair_truth_eval is not None  # nosec B101 - narrows Optional for mypy
    recovery_metrics = _row_metrics(
        pair_truth_eval_row(recovery_result.pair_truth_eval, POPULATION_UNIVERSE)
    )

    return baseline_metrics, recovery_metrics


def add_measurement_args(parser: argparse.ArgumentParser) -> None:
    """The workspace, sampling and scoring arguments this measurement and the
    short-name one share, which `measurement_settings` reads back."""
    add_workspace_roots_args(parser)
    parser.add_argument("--source-sample", type=int, default=5000)
    parser.add_argument("--distractor-sample", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.2)
    add_dry_run_arg(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_measurement_args(parser)
    return parser


def measurement_settings(args: argparse.Namespace) -> dict[str, object]:
    """The settings that decide this measurement, and so key its report."""
    return {
        "source_sample": args.source_sample,
        "distractor_sample": args.distractor_sample,
        "seed": args.seed,
        "top_k": args.top_k,
        "min_similarity": args.min_similarity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    settings = measurement_settings(args)
    report_path = resolve_measurement_report_path(
        roots, measurement=_MEASUREMENT, settings=settings
    )

    if args.dry_run:
        report_dry_run("measure_name_variant_recall", data_dir=roots.data, **settings)
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

    sampled_source, target_frame, variant_frame = build_sample_corpus(
        roots=roots,
        source_sample=args.source_sample,
        distractor_sample=args.distractor_sample,
        seed=args.seed,
    )
    print(
        f"corpus: {sampled_source.height} matched source rows, "
        f"{target_frame.height} target rows "
        f"({sampled_source.get_column('match_uri').n_unique()} true targets + "
        f"distractors), {variant_frame.height} name-variant sidecar rows in corpus"
    )

    with scratch_workspace(roots, prefix=_MEASUREMENT) as tmp_roots:
        write_corpus(
            tmp_roots,
            sampled_source=sampled_source,
            target_frame=target_frame,
            variant_frame=variant_frame,
        )
        baseline, self_recovery = run_comparison(
            tmp_roots, top_k=args.top_k, min_similarity=args.min_similarity
        )

    # Two separate labelled runs, printed side by side for
    # convenience -- never a delta between them, since a cross-system
    # match-recall run and a self-referential recovery run answer different
    # questions and are not comparable metric-for-metric.
    print("-" * 70)
    print(f"{'metric':<16}{'baseline':>16}{'self_recovery':>18}")
    for key in ("tp", "fp", "fn", "recall", "precision", "predicted_pairs"):
        b, r = baseline[key], self_recovery[key]
        if isinstance(b, float):
            print(f"{key:<16}{b:>16.4f}{r:>18.4f}")
        else:
            print(f"{key:<16}{b:>16}{r:>18}")

    payload = {
        "settings": settings,
        "baseline": baseline,
        "self_recovery": self_recovery,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote full measurement to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
