from __future__ import annotations

import polars as pl

from analysis.io_contract import (
    build_contract,
    profile_data_contract,
    write_schema_profile_artifacts,
)
from analysis.run_manifest import load_run_manifest
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_system_data(
    roots: WorkspaceRoots, system: str, rows: list[dict[str, object]]
) -> None:
    target = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / f"{system}-001.parquet")


def test_profile_data_contract_reports_required_and_optional_columns(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gb",
        [
            {
                "country_tokens": ["ACME", "LTD"],
                "global_tokens": ["ACME", "LTD"],
            }
        ],
    )
    _write_system_data(
        workspace_roots,
        "ie",
        [
            {
                "country_tokens": ["BETA", "LIMITED"],
            }
        ],
    )

    profile_df, drift_notes = profile_data_contract(
        roots=workspace_roots, systems=["gb", "ie"]
    )

    assert profile_df.height > 0
    required_rows = profile_df.filter(pl.col("required"))
    assert required_rows.height == 4
    assert required_rows.filter(~pl.col("present")).height == 1
    assert any("missing required column" in note for note in drift_notes)


def test_profile_data_contract_flags_missing_required_columns(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "fr",
        [
            {
                "country_tokens": ["ALPHA"],
            }
        ],
    )

    profile_df, drift_notes = profile_data_contract(
        roots=workspace_roots, systems=["fr"]
    )

    missing_required = profile_df.filter(pl.col("required") & ~pl.col("present"))
    assert missing_required.height == 1
    assert any(
        "missing required column 'global_tokens'" in note for note in drift_notes
    )


def test_write_schema_profile_artifacts_writes_parquet_and_markdown(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gleif",
        [
            {
                "country_tokens": ["OMEGA"],
                "global_tokens": ["OMEGA"],
            },
            {
                "country_tokens": ["SIGMA"],
                "global_tokens": ["SIGMA"],
            },
        ],
    )

    contract = build_contract(
        required_columns=["country_tokens"], optional_columns=["global_tokens"]
    )
    profile_df, drift_notes = profile_data_contract(
        roots=workspace_roots, systems=["gleif"], contract=contract
    )

    parquet_path, markdown_path, manifest_path = write_schema_profile_artifacts(
        roots=workspace_roots,
        run_date="2026-06-21",
        profile_df=profile_df,
        drift_notes=drift_notes,
        contract=contract,
    )

    assert parquet_path.exists()
    assert markdown_path.exists()
    assert manifest_path.exists()

    loaded = pl.read_parquet(parquet_path)
    assert loaded.height == profile_df.height

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Schema Profile" in markdown
    assert "Required column failures" in markdown
    assert "country_tokens" in markdown

    manifest = load_run_manifest(workspace_roots, "2026-06-21")
    phase_entry = manifest["phases"]["phase0_schema_profile"]
    assert phase_entry["run_date"] == "2026-06-21"
    assert phase_entry["systems"] == ["gleif"]
    assert phase_entry["parameters"]["required_columns"] == ["country_tokens"]
    assert phase_entry["outputs"]["schema_profile_parquet"] == parquet_path.as_posix()
    assert phase_entry["row_counts"]["profile_rows"] == profile_df.height


def test_profile_data_contract_supports_optional_override_contract(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gb",
        [
            {
                "country_tokens": ["ACME"],
                "global_tokens": ["ACME"],
            }
        ],
    )

    contract = build_contract(
        required_columns=["country_tokens", "global_tokens"],
        optional_columns=["company_type"],
    )
    profile_df, drift_notes = profile_data_contract(
        roots=workspace_roots, systems=["gb"], contract=contract
    )

    required_missing = profile_df.filter(pl.col("required") & ~pl.col("present"))
    optional_missing = profile_df.filter(~pl.col("required") & ~pl.col("present"))

    assert required_missing.height == 0
    assert optional_missing.height == 1
    assert drift_notes == []
