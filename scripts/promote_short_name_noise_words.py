from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from cli_common import run_reporting_argument_errors
from company_cleanse.corpus_noise_words import (
    validate_short_name_noise_word_candidates_payload,
)

DEFAULT_SOURCE_DIR = files("company_tokenize.resources").joinpath(
    "noise_word_candidates"
)

DEFAULT_DEST = files("company_cleanse.resources").joinpath(
    "short_name_noise_word_candidates.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the already-promoted, company_tokenize-owned per-system "
            "noise-word candidate resources "
            "(packages/company_tokenize/src/company_tokenize/resources/"
            "noise_word_candidates/<system>.json) into a single checked-in "
            "company_cleanse reference file "
            "(resources/short_name_noise_word_candidates.json), preserving each "
            "system's/language's own candidate list under its own key (plus "
            "'global') rather than flattening them into one universal set. This is "
            "a static, checked-in copy -- company_cleanse never imports "
            "company_tokenize at runtime. Validates every source file's shape "
            "before writing anything, and only writes when --write is passed."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help=(
            "Directory of promoted company_tokenize noise_word_candidates "
            "<system>.json files to merge. Defaults to company_tokenize's own "
            "packaged resource directory."
        ),
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=(
            "Path to the packaged company_cleanse resource file to overwrite. "
            "Defaults to resources/short_name_noise_word_candidates.json."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Actually overwrite --dest with the merged, validated content. "
            "Without this flag, sources are only validated and summarized (dry "
            "run) -- nothing is written."
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


def build_merged_payload(*, source_dir: Path, generated_utc: str) -> dict[str, Any]:
    """Merge every `<system>.json` in `source_dir` into one systems-keyed payload."""
    source_paths = sorted(source_dir.glob("*.json"))
    if not source_paths:
        raise FileNotFoundError(
            f"No candidate noise_word_candidates JSON files found in {source_dir}."
        )

    systems: dict[str, Any] = {}
    for source_path in source_paths:
        candidate_payload = _load_json_object(
            source_path, label=f"candidate noise-word file ({source_path.name})"
        )
        system = candidate_payload.get("system")
        if not isinstance(system, str) or not system.strip():
            raise ValueError(
                f"{source_path} is missing a non-empty 'system' field; cannot key "
                "the merged payload."
            )
        normalized_system = system.strip().lower()
        candidates = candidate_payload.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError(f"{source_path} must include a 'candidates' list.")

        systems[normalized_system] = {
            "candidates": candidates,
            "selection": candidate_payload.get("selection"),
            "source_generated_utc": candidate_payload.get("generated_utc"),
        }

    return {
        "generated_utc": generated_utc,
        "source": (
            "Promoted from company_tokenize's packaged noise_word_candidates "
            "resources (packages/company_tokenize/src/company_tokenize/resources/"
            "noise_word_candidates/<system>.json) via "
            "scripts/promote_short_name_noise_words.py. See "
            "packages/company_cleanse/README.md."
        ),
        "systems": systems,
    }


def summarize_promotion(
    *, candidate_payload: dict[str, Any], current_payload: dict[str, Any] | None
) -> list[str]:
    candidate_systems = candidate_payload.get("systems", {})
    current_systems = (
        current_payload.get("systems", {}) if current_payload is not None else {}
    )

    lines: list[str] = []
    for system_key in sorted(candidate_systems):
        candidate_count = len(candidate_systems[system_key].get("candidates", []))
        current_entry = current_systems.get(system_key)
        if current_entry is None:
            lines.append(
                f"[info] system={system_key} candidates={candidate_count} "
                "(no matching system in the current file to diff against)"
            )
            continue
        current_count = len(current_entry.get("candidates", []))
        delta = candidate_count - current_count
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"[info] system={system_key} current={current_count} "
            f"candidate={candidate_count} ({sign}{delta})"
        )

    dropped = sorted(set(current_systems) - set(candidate_systems))
    if dropped:
        lines.append(
            f"[warn] systems present currently but absent from candidate: {dropped}"
        )
    return lines


def run_promotion(*, source_dir: str, dest: str, write: bool) -> int:
    source_dir_path = Path(source_dir).resolve()
    dest_path = Path(dest).resolve()

    merged_payload = build_merged_payload(
        source_dir=source_dir_path,
        generated_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    validate_short_name_noise_word_candidates_payload(
        merged_payload, source_label=str(dest_path)
    )
    print(
        f"[info] merged {len(merged_payload['systems'])} system(s) from {source_dir_path}"
    )

    current_payload: dict[str, Any] | None = None
    if dest_path.exists():
        current_payload = _load_json_object(
            dest_path, label="current packaged resource"
        )

    for line in summarize_promotion(
        candidate_payload=merged_payload, current_payload=current_payload
    ):
        print(line)

    if not write:
        print(
            f"[info] dry run only -- nothing written. Re-run with --write to promote "
            f"{source_dir_path} -> {dest_path}."
        )
        return 0

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(merged_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[info] promoted: {source_dir_path} -> {dest_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_promotion(source_dir=args.source_dir, dest=args.dest, write=args.write)


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
