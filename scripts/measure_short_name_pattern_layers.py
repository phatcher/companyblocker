"""Measure each short-name derivation layer's real recovery rate, per language.

Real, reusable measurement tool for `company_cleanse.short_name_candidates`'s
layered deterministic short-name/significant-token candidate derivation --
not a throwaway analysis pass. Rerunning this script is how a later change (a
new layer, a wider extracted-language set, a decompounding tweak) gets
re-measured against real data rather than trusted on priors.

Two real, independent same-entity name-pair sources:

- Wikidata's `official` -> `short` name-variant sidecar
  (`data/wikidata/canonical/*/names/`), paired within an entity by matching
  `language_code` so a French official name is never checked against a
  German short name.
- FR SIRENE's `usual_name` -> `acronym` name-variant sidecar
  (`data/fr/canonical/*/names/`) joined back to the FR primary entity's own
  `name` column via `system_uri`, as an independent-source check against a
  real `sigleUniteLegale` field, not just a held-out Wikidata split.

Splits are entity-level (by the shared entity id -- `id` for Wikidata,
`system_uri` for FR), never row-level: an entity contributing more than one
language pair stays entirely on one side of the split, via a stable hash of
`(seed, entity_id)` so a rerun with the same seed reproduces the same split.

Usage:
    .venv/Scripts/python.exe scripts/measure_short_name_pattern_layers.py
    .venv/Scripts/python.exe scripts/measure_short_name_pattern_layers.py \
        --json-out artifacts/short-name-pattern-layer-yield.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
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
from company_cleanse._cli_helper import SETTINGS as COMPANY_CLEANSE_SETTINGS
from company_cleanse.short_name_candidates import (
    LAYERS,
    DecompoundFn,
    derive_short_name_candidate,
)

from workspace.data_layout import CANONICAL_LAYER_NAME, system_layer_dir
from workspace.layer_layout import resolve_primary_files
from workspace.roots import WorkspaceRoots

_RE_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9 ]+")
_RE_MULTI_SPACE = re.compile(r"\s+")

_DECLARATIONS = declared_settings(COMPANY_CLEANSE_SETTINGS)
_SURFACE = {
    "short_name_layer_seed": {"flag": "--seed"},
    "short_name_layer_german_decompounding": {"flag": "--with-german-decompounding"},
}

_WIKIDATA_LANGUAGES = ("de", "fr", "en")
_SPLIT_NAMES = ("train", "test", "validate")
_DEFAULT_SPLIT_WEIGHTS = (0.7, 0.15, 0.15)


def normalize_for_comparison(value: str | None) -> str:
    """Lowercase, ASCII-punctuation-stripped, single-spaced form for equality checks."""
    if not value:
        return ""
    lowered = value.strip().lower()
    stripped = _RE_NON_ALNUM_SPACE.sub("", lowered)
    return _RE_MULTI_SPACE.sub(" ", stripped).strip()


def entity_split(
    entity_ids: Iterable[str],
    *,
    seed: int = 0,
    weights: tuple[float, float, float] = _DEFAULT_SPLIT_WEIGHTS,
) -> dict[str, str]:
    """Deterministically assign each entity id to train/test/validate.

    Uses a stable hash of `(seed, entity_id)` rather than a random shuffle,
    so a rerun with the same seed reproduces the same split even as new
    entities are added -- an entity's split assignment never depends on
    which other entities happen to be present in the same run.
    """
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"Split weights must sum to 1.0, got {weights!r}.")

    cumulative = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running)

    assignment: dict[str, str] = {}
    for entity_id in set(entity_ids):
        digest = hashlib.sha256(f"{seed}:{entity_id}".encode()).hexdigest()
        # First 8 hex chars -> a stable float in [0, 1).
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        for split_name, threshold in zip(_SPLIT_NAMES, cumulative):
            if bucket < threshold:
                assignment[entity_id] = split_name
                break
        else:
            assignment[entity_id] = _SPLIT_NAMES[-1]
    return assignment


def _find_latest_snapshot(roots: WorkspaceRoots, system: str) -> Path:
    canonical_dir = system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME)
    snapshots = sorted(p for p in canonical_dir.glob("*") if p.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"No canonical snapshots found under {canonical_dir}.")
    return snapshots[-1]


def load_wikidata_pairs(
    roots: WorkspaceRoots, *, languages: tuple[str, ...] = _WIKIDATA_LANGUAGES
) -> pl.DataFrame:
    """Real Wikidata `official` -> `short` same-entity pairs, matched by language."""
    names_dir = _find_latest_snapshot(roots, "wikidata") / "names"
    files = sorted(names_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Wikidata name-variant parquet files at {names_dir}."
        )

    names = pl.scan_parquet([str(f) for f in files]).select(
        ["id", "name_type", "language_code", "name"]
    )
    official = (
        names.filter(pl.col("name_type") == "official")
        .filter(pl.col("language_code").is_in(list(languages)))
        .select(["id", "language_code", pl.col("name").alias("official_name")])
    )
    short = (
        names.filter(pl.col("name_type") == "short")
        .filter(pl.col("language_code").is_in(list(languages)))
        .select(["id", "language_code", pl.col("name").alias("short_name")])
    )
    pairs = (
        official.join(short, on=["id", "language_code"], how="inner")
        .rename({"id": "entity_id"})
        .with_columns(pl.lit("wikidata").alias("source"))
        .collect()
    )
    return pairs


def load_fr_pairs(roots: WorkspaceRoots) -> pl.DataFrame:
    """Real FR `name` -> `acronym` (`sigleUniteLegale`) pairs, independent of Wikidata."""
    snapshot = _find_latest_snapshot(roots, "fr")
    names_files = sorted((snapshot / "names").glob("*.parquet"))
    primary_files = resolve_primary_files(snapshot, system_code="fr")
    if not names_files or not primary_files:
        raise FileNotFoundError(f"FR canonical data incomplete under {snapshot}.")

    acronyms = (
        pl.scan_parquet([str(f) for f in names_files])
        .filter(pl.col("name_type") == "acronym")
        .select(["system_uri", pl.col("name").alias("short_name")])
    )
    primary = pl.scan_parquet([str(f) for f in primary_files]).select(
        ["system_uri", pl.col("name").alias("official_name")]
    )
    pairs = (
        acronyms.join(primary, on="system_uri", how="inner")
        .rename({"system_uri": "entity_id"})
        .with_columns(
            pl.lit("fr").alias("language_code"), pl.lit("fr_acronym").alias("source")
        )
        .collect()
    )
    return pairs


@dataclass
class LayerYield:
    source: str
    split: str
    language_code: str
    layer: str
    n_pairs: int = 0
    n_fired: int = 0
    n_correct: int = 0

    @property
    def coverage(self) -> float:
        return self.n_fired / self.n_pairs if self.n_pairs else 0.0

    @property
    def recall(self) -> float:
        return self.n_correct / self.n_pairs if self.n_pairs else 0.0

    @property
    def precision(self) -> float:
        return self.n_correct / self.n_fired if self.n_fired else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "split": self.split,
            "language_code": self.language_code,
            "layer": self.layer,
            "n_pairs": self.n_pairs,
            "n_fired": self.n_fired,
            "n_correct": self.n_correct,
            "coverage": round(self.coverage, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
        }


_SUMMARY_ROW_NAMES = ("default_order", "oracle_any_layer")


def measure_pairs(
    pairs: pl.DataFrame,
    *,
    split_map: dict[str, str],
    decompound_fn_by_language: dict[str, DecompoundFn] | None = None,
) -> list[LayerYield]:
    """Per (split, language, layer) coverage/recall/precision, plus two summary rows.

    `default_order` is the real number a caller of `derive_short_name_candidate()`
    (its own default `enabled_layers` order) actually gets: one candidate per
    name, whichever layer's turn came up first. `oracle_any_layer` is a
    distinct, more optimistic ceiling -- would *any* layer, run alone, have
    produced the right answer -- useful for judging how much headroom a
    smarter combination strategy could still capture, but not a number a
    real single-candidate caller receives.
    """
    decompound_fn_by_language = decompound_fn_by_language or {}
    layer_names = (*LAYERS, *_SUMMARY_ROW_NAMES)
    results: dict[tuple[str, str, str, str], LayerYield] = {}

    def _entry(split_name: str, language_code: str, layer_name: str) -> LayerYield:
        key = (source, split_name, language_code, layer_name)
        return results.setdefault(
            key,
            LayerYield(
                source=source,
                split=split_name,
                language_code=language_code,
                layer=layer_name,
            ),
        )

    source = pairs["source"][0] if pairs.height else "unknown"
    for entity_id, language_code, official_name, short_name in zip(
        pairs["entity_id"].to_list(),
        pairs["language_code"].to_list(),
        pairs["official_name"].to_list(),
        pairs["short_name"].to_list(),
    ):
        split_name = split_map.get(entity_id, "train")
        target = normalize_for_comparison(short_name)
        decompound_fn = decompound_fn_by_language.get(language_code)

        any_layer_hit = False
        for layer_name in LAYERS:
            entry = _entry(split_name, language_code, layer_name)
            entry.n_pairs += 1

            candidate = derive_short_name_candidate(
                official_name,
                enabled_layers=(layer_name,),
                decompound_fn=decompound_fn,
            )
            if candidate is None:
                continue
            entry.n_fired += 1
            if target and normalize_for_comparison(candidate.value) == target:
                entry.n_correct += 1
                any_layer_hit = True

        default_entry = _entry(split_name, language_code, "default_order")
        default_entry.n_pairs += 1
        default_candidate = derive_short_name_candidate(
            official_name, decompound_fn=decompound_fn
        )
        if default_candidate is not None:
            default_entry.n_fired += 1
            if target and normalize_for_comparison(default_candidate.value) == target:
                default_entry.n_correct += 1

        oracle_entry = _entry(split_name, language_code, "oracle_any_layer")
        oracle_entry.n_pairs += 1
        if any_layer_hit:
            oracle_entry.n_fired += 1
            oracle_entry.n_correct += 1

    ordered_keys = sorted(
        results,
        key=lambda k: (
            k[0],
            _SPLIT_NAMES.index(k[1]) if k[1] in _SPLIT_NAMES else 99,
            k[2],
            layer_names.index(k[3]),
        ),
    )
    return [results[key] for key in ordered_keys]


def print_report(results: list[LayerYield]) -> None:
    print(
        f"{'source':<12}{'split':<10}{'lang':<6}{'layer':<28}"
        f"{'n':>8}{'coverage':>10}{'recall':>9}{'precision':>11}"
    )
    for entry in results:
        print(
            f"{entry.source:<12}{entry.split:<10}{entry.language_code:<6}{entry.layer:<28}"
            f"{entry.n_pairs:>8}{entry.coverage:>10.2%}{entry.recall:>9.2%}"
            f"{entry.precision:>11.2%}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_roots_args(parser)
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Optional path to write the full measurement as JSON. A relative "
            "path resolves against the checkout, never the working directory."
        ),
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the seed, decompounding choice and --json-out "
            "destination this run would use, without reading any data or "
            "writing anything."
        ),
    )
    return parser


def _resolve_json_out(json_out: str | None, roots: WorkspaceRoots) -> Path | None:
    if not json_out:
        return None
    candidate = Path(json_out)
    return (
        candidate if candidate.is_absolute() else (roots.checkout / candidate).resolve()
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    values = resolved_setting_values(resolved)
    seed = int(values["short_name_layer_seed"])  # type: ignore[arg-type]
    with_german_decompounding = bool(values["short_name_layer_german_decompounding"])
    json_out_path = _resolve_json_out(args.json_out, roots)

    if args.dry_run:
        report_resolved_settings(
            "measure_short_name_pattern_layers",
            resolved,
            wikidata_canonical_dir=system_layer_dir(
                roots, "wikidata", layer=CANONICAL_LAYER_NAME
            ),
            fr_canonical_dir=system_layer_dir(roots, "fr", layer=CANONICAL_LAYER_NAME),
            json_out=json_out_path,
        )
        if json_out_path is not None:
            report_output_plan(
                "[dry-run]",
                [
                    PlannedOutput(
                        json_out_path, OUTPUT_CLEAR, note="rewritten from scratch"
                    )
                ],
            )
        return 0

    decompound_fn_by_language: dict[str, DecompoundFn] = {}
    if with_german_decompounding:
        from analysis.german_decompound import split_compound

        decompound_fn_by_language["de"] = split_compound

    wikidata_pairs = load_wikidata_pairs(roots)
    fr_pairs = load_fr_pairs(roots)

    wikidata_split = entity_split(wikidata_pairs["entity_id"].to_list(), seed=seed)
    fr_split = entity_split(fr_pairs["entity_id"].to_list(), seed=seed)

    results = measure_pairs(
        wikidata_pairs,
        split_map=wikidata_split,
        decompound_fn_by_language=decompound_fn_by_language,
    ) + measure_pairs(
        fr_pairs,
        split_map=fr_split,
        decompound_fn_by_language=decompound_fn_by_language,
    )

    print(
        f"wikidata pairs: {wikidata_pairs.height} "
        f"({wikidata_pairs['entity_id'].n_unique()} entities); "
        f"fr pairs: {fr_pairs.height} ({fr_pairs['entity_id'].n_unique()} entities)"
    )
    print(f"decompounding: {'on' if decompound_fn_by_language else 'off'}")
    print("-" * 90)
    print_report(results)

    if json_out_path is not None:
        payload = {
            "decompounding_enabled": bool(decompound_fn_by_language),
            "seed": seed,
            "wikidata_pair_count": wikidata_pairs.height,
            "fr_pair_count": fr_pairs.height,
            "results": [entry.as_dict() for entry in results],
        }
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote full measurement to {json_out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
