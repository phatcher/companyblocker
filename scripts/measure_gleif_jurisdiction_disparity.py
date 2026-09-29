"""Measure GLEIF's pooled tokenizer for per-jurisdiction fertility/UNK-rate disparity.

Read-only analysis script, not a build task. GLEIF's tokenizer was trained
as one flat system (`artifacts/tokenizers/work/gleif/wordpiece/model.json`,
vocab=40000, min_frequency=1 -- the naive baseline, since no optimize
candidate ever cleared the safety gates for this scope), even though
GLEIF's own documentation describes it as "not a country registry -- a
multi-hive, global reference dataset" spanning many `jurisdiction_code`
values. This script loads that already-trained pooled tokenizer and
evaluates it, per `jurisdiction_code`, against the real GLEIF cleansed
population (`data/gleif/cleansed/jurisdiction_code=*/`) -- reusing the
exact fertility/unk-rate formulas `evaluate_tokenizer_metrics`
(`company_tokenize.optimize`) uses for the pooled/per-system training
report, so the numbers here are directly comparable to that existing report
rather than a new, incompatible metric. The output says whether one or more
jurisdictions are served systematically worse than the pooled average, as
evidence for whether a per-jurisdiction retrain is justified. Does not
train, retrain, or write any tokenizer artifact.

Usage:
    .venv/Scripts/python.exe scripts/measure_gleif_jurisdiction_disparity.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import _bootstrap
import polars as pl
from cli_common import run_reporting_argument_errors
from company_tokenize.paths import tokenizer_directory_files, tokenizer_id
from company_tokenize.training import load_tokenizer_encoder

from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.layer_layout import resolve_primary_files
from workspace.roots import WorkspaceRoots, default_workspace_roots
from workspace.tokenizer_store import promoted_tokenizer_directory

DEFAULT_SYSTEM = "gleif"
DEFAULT_TRAINER = "wordpiece"
DEFAULT_NAME_COL = "name_cleansed"
DEFAULT_MIN_ROWS = 1000

# Same gate constants `OPTIMIZE_BASE_DEFAULTS` uses for country scopes
# (`src/training/optimize_execution.py`) -- reused here as the "meaningful"
# bar for a jurisdiction's disparity, not a new threshold invented for this
# measurement.
FERTILITY_TARGET = 1.25
FERTILITY_TOLERANCE = 0.10
UNK_RATE_THRESHOLD = 0.005


def _token_character_length(token: str) -> int:
    # Mirrors `evaluate_tokenizer_metrics`'s subword-marker normalization.
    if token.startswith("##"):
        return len(token[2:])
    if token.startswith("▁"):  # sentencepiece "▁" marker
        return len(token[1:])
    return len(token)


@dataclass
class JurisdictionFertility:
    jurisdiction: str
    rows: int = 0
    tokens: int = 0
    unk_tokens: int = 0
    single_char_tokens: int = 0
    fertility_sum: float = 0.0
    token_count_sum: int = 0

    @property
    def unk_rate(self) -> float:
        return (self.unk_tokens / self.tokens) if self.tokens else 0.0

    @property
    def fertility(self) -> float:
        return (self.fertility_sum / self.rows) if self.rows else 0.0

    @property
    def fertility_distance(self) -> float:
        return abs(self.fertility - FERTILITY_TARGET)

    @property
    def single_char_token_pct(self) -> float:
        return ((self.single_char_tokens / self.tokens) * 100.0) if self.tokens else 0.0

    @property
    def token_count_mean(self) -> float:
        return (self.token_count_sum / self.rows) if self.rows else 0.0

    @property
    def breaches_fertility_tolerance(self) -> bool:
        return self.fertility_distance > FERTILITY_TOLERANCE

    @property
    def breaches_unk_rate_threshold(self) -> bool:
        return self.unk_rate > UNK_RATE_THRESHOLD

    def as_dict(self) -> dict[str, object]:
        return {
            "jurisdiction": self.jurisdiction,
            "rows": self.rows,
            "unk_rate": round(self.unk_rate, 6),
            "single_char_token_pct": round(self.single_char_token_pct, 4),
            "fertility": round(self.fertility, 6),
            "fertility_distance": round(self.fertility_distance, 6),
            "token_count_mean": round(self.token_count_mean, 4),
            "breaches_fertility_tolerance": self.breaches_fertility_tolerance,
            "breaches_unk_rate_threshold": self.breaches_unk_rate_threshold,
        }


def measure_jurisdiction_disparity(
    *,
    roots: WorkspaceRoots,
    system: str,
    tokenizer_path: Path,
    trainer: str,
    name_col: str,
) -> dict[str, JurisdictionFertility]:
    encode_tokens, unk_token = load_tokenizer_encoder(
        tokenizer_path=tokenizer_path, trainer=trainer
    )

    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    partition_files = resolve_primary_files(cleansed_dir, system_code=system)
    if not partition_files:
        raise FileNotFoundError(
            f"No jurisdiction_code=*/*.parquet files found under {cleansed_dir}."
        )

    df = pl.read_parquet(
        [str(f) for f in partition_files], columns=["jurisdiction_code", name_col]
    )

    buckets: dict[str, JurisdictionFertility] = {}
    for row in df.iter_rows(named=True):
        name_raw = row.get(name_col)
        name = "" if name_raw is None else str(name_raw).strip()
        if not name:
            continue

        jurisdiction_raw = row.get("jurisdiction_code")
        jurisdiction = (
            "unknown"
            if jurisdiction_raw is None
            else str(jurisdiction_raw).strip().lower()
        )
        if not jurisdiction:
            jurisdiction = "unknown"

        tokens = encode_tokens(name)
        token_count = len(tokens)
        if token_count == 0:
            continue

        word_count = max(len(name.split()), 1)
        unk_count = sum(1 for token in tokens if token == unk_token)
        single_char_count = sum(
            1 for token in tokens if _token_character_length(token) == 1
        )
        fertility = token_count / word_count

        bucket = buckets.setdefault(jurisdiction, JurisdictionFertility(jurisdiction))
        bucket.rows += 1
        bucket.tokens += token_count
        bucket.unk_tokens += unk_count
        bucket.single_char_tokens += single_char_count
        bucket.fertility_sum += fertility
        bucket.token_count_sum += token_count

    return buckets


def _pooled(buckets: dict[str, JurisdictionFertility]) -> JurisdictionFertility:
    pooled = JurisdictionFertility("__pooled__")
    for bucket in buckets.values():
        pooled.rows += bucket.rows
        pooled.tokens += bucket.tokens
        pooled.unk_tokens += bucket.unk_tokens
        pooled.single_char_tokens += bucket.single_char_tokens
        pooled.fertility_sum += bucket.fertility_sum
        pooled.token_count_sum += bucket.token_count_sum
    return pooled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-jurisdiction fertility/UNK-rate disparity in "
            "GLEIF's existing pooled tokenizer (evaluated against "
            "data/gleif/cleansed/, grouped by jurisdiction_code). Read-only; "
            "does not train or write any tokenizer artifact."
        )
    )
    parser.add_argument(
        "--root",
        default=str(_bootstrap.REPO_ROOT),
        help="Project root containing data/.",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--tokenizer", default=DEFAULT_TRAINER)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Path (relative to --root, or absolute) to a tokenizer model. "
            "Defaults to the tokenizer promoted for --system and --tokenizer."
        ),
    )
    parser.add_argument("--name-col", default=DEFAULT_NAME_COL)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help=(
            "Minimum row count for a jurisdiction to be included in the "
            "disparity table/verdict -- small partitions are reported "
            "separately, not folded into the headline numbers."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many qualifying jurisdictions (worst fertility_distance first) to print.",
    )
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)
    if args.tokenizer_path is None:
        tokenizer_path = tokenizer_directory_files(
            promoted_tokenizer_directory(
                roots,
                system=args.system,
                tokenizer_id=tokenizer_id(args.tokenizer),
            ),
            trainer=args.tokenizer,
        ).model
    else:
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.is_absolute():
            tokenizer_path = root / tokenizer_path

    buckets = measure_jurisdiction_disparity(
        roots=roots,
        system=args.system,
        tokenizer_path=tokenizer_path,
        trainer=args.tokenizer,
        name_col=args.name_col,
    )
    pooled = _pooled(buckets)

    qualifying = {j: b for j, b in buckets.items() if b.rows >= args.min_rows}
    small = {j: b for j, b in buckets.items() if b.rows < args.min_rows}
    small_rows = sum(b.rows for b in small.values())

    print(
        f"system={args.system} trainer={args.tokenizer} tokenizer_path={tokenizer_path}"
    )
    print(
        f"pooled: rows={pooled.rows} unk_rate={pooled.unk_rate:.6f} "
        f"fertility={pooled.fertility:.6f} "
        f"fertility_distance={pooled.fertility_distance:.6f} "
        f"single_char_token_pct={pooled.single_char_token_pct:.4f}% "
        f"token_count_mean={pooled.token_count_mean:.4f}"
    )
    print(
        f"jurisdictions: {len(buckets)} total, {len(qualifying)} with "
        f">= {args.min_rows} rows ({sum(b.rows for b in qualifying.values())} rows), "
        f"{len(small)} below threshold ({small_rows} rows, folded out of the table below)"
    )

    ranked = sorted(
        qualifying.values(), key=lambda b: b.fertility_distance, reverse=True
    )

    print()
    print(
        f"{'jurisdiction':<14}{'rows':>10}{'unk_rate':>12}{'fertility':>12}"
        f"{'fert_dist':>12}{'single_char%':>14}{'flags':>16}"
    )
    for bucket in ranked[: args.top_n]:
        flags = []
        if bucket.breaches_fertility_tolerance:
            flags.append("FERT")
        if bucket.breaches_unk_rate_threshold:
            flags.append("UNK")
        flag_str = ",".join(flags) if flags else "-"
        print(
            f"{bucket.jurisdiction:<14}{bucket.rows:>10}{bucket.unk_rate:>12.6f}"
            f"{bucket.fertility:>12.6f}{bucket.fertility_distance:>12.6f}"
            f"{bucket.single_char_token_pct:>13.4f}%{flag_str:>16}"
        )

    breaching_fertility = [
        b for b in qualifying.values() if b.breaches_fertility_tolerance
    ]
    breaching_unk = [b for b in qualifying.values() if b.breaches_unk_rate_threshold]
    breaching_fertility_rows = sum(b.rows for b in breaching_fertility)
    breaching_unk_rows = sum(b.rows for b in breaching_unk)
    qualifying_rows = sum(b.rows for b in qualifying.values()) or 1

    print()
    print(
        f"gate thresholds: fertility_tolerance={FERTILITY_TOLERANCE} "
        f"(target={FERTILITY_TARGET}), unk_rate_threshold={UNK_RATE_THRESHOLD}"
    )
    print(
        f"jurisdictions breaching fertility_tolerance: {len(breaching_fertility)}"
        f"/{len(qualifying)} ({breaching_fertility_rows} rows, "
        f"{breaching_fertility_rows / qualifying_rows * 100:.4f}% of qualifying rows) "
        f"while pooled fertility_distance={pooled.fertility_distance:.6f} is "
        f"{'within' if not pooled.breaches_fertility_tolerance else 'also outside'} tolerance"
    )
    print(
        f"jurisdictions breaching unk_rate_threshold: {len(breaching_unk)}"
        f"/{len(qualifying)} ({breaching_unk_rows} rows, "
        f"{breaching_unk_rows / qualifying_rows * 100:.4f}% of qualifying rows) "
        f"while pooled unk_rate={pooled.unk_rate:.6f} is "
        f"{'within' if not pooled.breaches_unk_rate_threshold else 'also outside'} threshold"
    )

    if args.json_out:
        payload = {
            "system": args.system,
            "tokenizer": args.tokenizer,
            "tokenizer_path": str(tokenizer_path),
            "fertility_target": FERTILITY_TARGET,
            "fertility_tolerance": FERTILITY_TOLERANCE,
            "unk_rate_threshold": UNK_RATE_THRESHOLD,
            "min_rows": args.min_rows,
            "pooled": pooled.as_dict(),
            "jurisdictions_total": len(buckets),
            "jurisdictions_qualifying": len(qualifying),
            "small_jurisdictions_folded_out": len(small),
            "small_jurisdictions_rows": small_rows,
            "jurisdictions": [b.as_dict() for b in ranked],
            "breaching_fertility_tolerance_count": len(breaching_fertility),
            "breaching_fertility_tolerance_rows": breaching_fertility_rows,
            "breaching_unk_rate_threshold_count": len(breaching_unk),
            "breaching_unk_rate_threshold_rows": breaching_unk_rows,
        }
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote full measurement to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
