from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import run_reporting_argument_errors

from acquisition.catalog import load_catalog_plans
from acquisition.catalog.validator import validate_catalog_schema


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate acquisition catalog schema and semantic constraints."
    )
    parser.add_argument(
        "--scan",
        default=".",
        help="Project root containing acquisition/catalog/.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.scan).resolve()
    systems_dir = root / "src" / "acquisition" / "catalog" / "systems"
    schema_path = root / "src" / "acquisition" / "catalog" / "systems.schema.json"

    validated_files = validate_catalog_schema(
        systems_dir=systems_dir, schema_path=schema_path
    )
    country, system = load_catalog_plans(systems_dir=systems_dir)

    print(f"Validated {len(validated_files)} catalog files")
    print(f"  country plans: {len(country)}")
    print(f"  system plans: {len(system)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
