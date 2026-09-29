from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from cli_common import run_reporting_argument_errors
from company_cleanse.config import (
    summarize_profiled_noise_word_counts,
    validate_profiled_noise_words_payload,
)

DEFAULT_DEST = Path(
    str(files("company_cleanse.resources").joinpath("noise_words.json"))
)
"""The packaged resource file, resolved through `importlib.resources` rather
than a repository-root-relative path: this is a package resource, not a
`data/` or `artifacts/` location `workspace` owns."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a validated, corpus-derived noise-word candidate (as written by "
            "scripts/generate_noise_words.py to artifacts/tokenizers/work/<system>/noise_words.json) "
            "to the packaged company_cleanse default (noise_words.json). Validates the "
            "candidate's 'profiles' shape against what "
            "company_cleanse.config.get_profiled_noise_words() expects before writing "
            "anything, and only writes when --write is passed -- this is a deliberate, "
            "reviewed promotion, not an automatic one."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Path to the candidate noise_words.json to promote "
            "(for example artifacts/tokenizers/work/gb/noise_words.json)."
        ),
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=(
            "Path to the packaged resource file to overwrite. Defaults to the packaged "
            "company_cleanse noise_words.json."
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


def summarize_promotion(
    *, candidate_payload: dict[str, Any], current_payload: dict[str, Any] | None
) -> list[str]:
    """Build human-readable per-profile token-count diff lines."""
    candidate_counts = summarize_profiled_noise_word_counts(candidate_payload)
    current_counts = (
        summarize_profiled_noise_word_counts(current_payload)
        if current_payload is not None
        else {}
    )

    lines: list[str] = []
    for profile_key, candidate_count in candidate_counts.items():
        current_count = current_counts.get(profile_key)
        if current_count is None:
            lines.append(
                f"[info] profile={profile_key} candidate={candidate_count} "
                "(no matching profile in the current file to diff against)"
            )
        else:
            delta = candidate_count - current_count
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"[info] profile={profile_key} current={current_count} "
                f"candidate={candidate_count} ({sign}{delta})"
            )
    return lines


def run_promotion(*, source: str, dest: str, write: bool) -> int:
    source_path = Path(source).resolve()
    dest_path = Path(dest).resolve()

    candidate_payload = _load_json_object(
        source_path, label="candidate noise-word file"
    )
    validate_profiled_noise_words_payload(
        candidate_payload, source_label=str(source_path)
    )
    print(f"[info] candidate validated: {source_path}")

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
