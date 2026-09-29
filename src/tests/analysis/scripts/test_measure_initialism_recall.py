from __future__ import annotations

import sys

import polars as pl

from scripts import measure_initialism_recall
from scripts.measure_initialism_recall import (
    _discover_jurisdictions,
    measure_jurisdiction,
)
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots

# match_uri is "<target_system>://<id>", the target's own system_uri value
# verbatim (see _resolve_target_system's docstring).
_TARGET_SYSTEM_URI = "gb://gb:1"


def _write_matched_partition(
    roots: WorkspaceRoots, *, system: str, country: str
) -> None:
    partition_dir = layer_partition_dir(
        system_layer_dir(roots, system, layer=MATCHED_LAYER_NAME), value=country
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"{system}:1"],
            "match_uri": [_TARGET_SYSTEM_URI],
        }
    ).write_parquet(partition_dir / "part-00001.parquet")
    # `matched/` carries no cleansed name: the source's forms sit in its own
    # cleansed layer, keyed by `system_uri`.
    cleansed_dir = layer_partition_dir(
        system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME), value=country
    )
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"{system}:1"],
            "short_name": ["ibm"],
            "name_cleansed": ["ibm"],
        }
    ).write_parquet(cleansed_dir / "part-00001.parquet")


def _write_cleansed_partition(
    roots: WorkspaceRoots, *, system: str, country: str
) -> None:
    partition_dir = layer_partition_dir(
        system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME), value=country
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [_TARGET_SYSTEM_URI],
            "short_name": ["international business machines"],
            "name_cleansed": ["international business machines"],
        }
    ).write_parquet(partition_dir / "part-00001.parquet")


def test_discover_jurisdictions_finds_family_split_primary_directory(
    workspace_roots: WorkspaceRoots,
):
    """Real matched/ output lives under a `primary/` family directory, not
    directly at the layer's own top level.
    """
    _write_matched_partition(workspace_roots, system="gleif", country="gb")

    jurisdictions = _discover_jurisdictions(
        system_layer_dir(workspace_roots, "gleif", layer=MATCHED_LAYER_NAME)
    )

    assert jurisdictions == ["gb"]


def test_measure_jurisdiction_reads_family_split_primary_directories(
    workspace_roots: WorkspaceRoots,
):
    _write_matched_partition(workspace_roots, system="gleif", country="gb")
    _write_cleansed_partition(workspace_roots, system="gb", country="gb")

    result = measure_jurisdiction(
        roots=workspace_roots, source_system="gleif", jurisdiction="gb"
    )

    assert result.target_system == "gb"
    assert result.matched_pairs == 1
    assert result.initialism_shaped_pairs == 1


def test_dry_run_reports_the_resolved_settings_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_initialism_recall.py",
            "--data-dir",
            str(workspace_roots.data),
            "--source",
            "gleif",
            "--json-out",
            "tmp/initialism.json",
            "--dry-run",
        ],
    )

    exit_code = measure_initialism_recall.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "source_system='gleif'" in out
    assert "would clear" in out
    assert "initialism.json" in out
