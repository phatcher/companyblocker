from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_FILE = Path(__file__).resolve().parent / "systems.schema.json"


def _default_systems_dir() -> Path:
    return Path(__file__).resolve().parent / "systems"


def load_catalog_schema(schema_path: Path | None = None) -> dict[str, object]:
    resolved = schema_path or SCHEMA_FILE
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_catalog_schema(
    systems_dir: Path | None = None,
    schema_path: Path | None = None,
) -> list[Path]:
    resolved_systems_dir = systems_dir or _default_systems_dir()
    schema = load_catalog_schema(schema_path)
    validator = Draft202012Validator(schema)

    if not resolved_systems_dir.exists():
        return []

    files = sorted(resolved_systems_dir.glob("*.json"))
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
        if errors:
            first = errors[0]
            loc = "/".join(str(part) for part in first.path)
            suffix = f" at '{loc}'" if loc else ""
            raise ValueError(
                f"Schema validation failed for {path}{suffix}: {first.message}"
            )

    return files
