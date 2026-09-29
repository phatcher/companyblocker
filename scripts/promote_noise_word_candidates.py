from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from cli_common import run_reporting_argument_errors
from company_tokenize import (
    summarize_noise_word_candidates,
    validate_noise_word_candidates_payload,
)

DEFAULT_DEST_DIR = Path(
    str(files("company_tokenize.resources").joinpath("noise_word_candidates"))
)
"""The packaged resource directory, resolved through `importlib.resources`
rather than a repository-root-relative path: this is a package resource, not
a `data/` or `artifacts/` location `workspace` owns."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a validated, corpus-derived noise-word candidate export (as "
            "written by scripts/export_noise_word_candidates.py to "
            "artifacts/tokenizers/work/<system>/noise_word_candidates.json) to "
            "company_tokenize's own packaged resources "
            "(src/company_tokenize/resources/noise_word_candidates/<system>.json). "
            "This is deliberately a company_tokenize-owned packaged artifact, not a "
            "write into company_cleanse; company_cleanse's own consumption path "
            "pulls this into its reference data. Validates the candidate's shape before writing anything, "
            "and only writes when --write is passed -- this is a deliberate, reviewed "
            "promotion, not an automatic one."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Path to the candidate noise_word_candidates.json to promote "
            "(for example artifacts/tokenizers/work/gb/noise_word_candidates.json)."
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        help=(
            "Path to the packaged resource file to overwrite. Defaults to "
            "src/company_tokenize/resources/noise_word_candidates/<system>.json, "
            "using the 'system' field recorded in --source."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Actually overwrite --dest with --source's validated content. Without this "
            "flag, the candidate is only validated and summarized (dry run) -- nothing "
            "is written."
        ),
    )
    return parser


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_dest_path(*, dest: str | None, candidate_payload: dict[str, Any]) -> Path:
    if dest:
        return Path(dest).resolve()
    system = candidate_payload.get("system")
    if not isinstance(system, str) or not system.strip():
        raise ValueError(
            "--dest was not given and --source's 'system' field is missing/empty; "
            "cannot derive a default destination."
        )
    return DEFAULT_DEST_DIR / f"{system.strip().lower()}.json"


def summarize_promotion(
    *, candidate_payload: dict[str, Any], current_payload: dict[str, Any] | None
) -> list[str]:
    """Build a human-readable candidate-count/idf-range diff against the current file."""
    candidate_summary = summarize_noise_word_candidates(candidate_payload)
    current_summary = (
        summarize_noise_word_candidates(current_payload)
        if current_payload is not None
        else None
    )

    if current_summary is None:
        return [
            (
                f"[info] candidates={candidate_summary['candidate_count']} "
                f"idf_range=[{candidate_summary['idf_min']}, "
                f"{candidate_summary['idf_max']}] "
                "(no current packaged file to diff against)"
            )
        ]

    delta = candidate_summary["candidate_count"] - current_summary["candidate_count"]
    sign = "+" if delta >= 0 else ""
    return [
        (
            f"[info] candidates: current={current_summary['candidate_count']} "
            f"candidate={candidate_summary['candidate_count']} ({sign}{delta})"
        ),
        (
            f"[info] idf_range: current=[{current_summary['idf_min']}, "
            f"{current_summary['idf_max']}] "
            f"candidate=[{candidate_summary['idf_min']}, {candidate_summary['idf_max']}]"
        ),
    ]


def run_promotion(*, source: str, dest: str | None, write: bool) -> int:
    source_path = Path(source).resolve()

    candidate_payload = _load_json_object(
        source_path, label="candidate noise-word export"
    )
    validated_payload = validate_noise_word_candidates_payload(
        candidate_payload, source_label=str(source_path)
    )
    print(f"[info] candidate validated: {source_path}")

    dest_path = _resolve_dest_path(dest=dest, candidate_payload=validated_payload)

    current_payload: dict[str, Any] | None = None
    if dest_path.exists():
        current_payload = _load_json_object(
            dest_path, label="current packaged resource"
        )

    for line in summarize_promotion(
        candidate_payload=candidate_payload, current_payload=current_payload
    ):
        print(line)

    if not write:
        print(
            f"[info] dry run only -- nothing written. Re-run with --write to promote "
            f"{source_path} over {dest_path}."
        )
        return 0

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(
        json.dumps(candidate_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[info] promoted: {source_path} -> {dest_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_promotion(source=args.source, dest=args.dest, write=args.write)


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
