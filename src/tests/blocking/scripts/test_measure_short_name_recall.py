from __future__ import annotations

import polars as pl
import pytest

from blocking.run_layout import resolve_measurement_report_path
from scripts.measure_short_name_recall import (
    _write_partition,
    build_sample_corpus,
    main,
)
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_roots_it_reads_and_writes_under_and_writes_nothing(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--temp-dir",
            str(workspace_roots.temp),
            "--top-k",
            "3",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    report = resolve_measurement_report_path(
        workspace_roots,
        measurement="short_name_recall",
        settings={
            "source_sample": 5000,
            "distractor_sample": 50000,
            "seed": 42,
            "top_k": 3,
            "min_similarity": 0.2,
        },
    )
    assert f"data_dir={workspace_roots.data!r}" in out
    assert f"would extend {workspace_roots.temp}" in out
    assert f"would clear {report}" in out
    assert not workspace_roots.artifacts.exists()
    assert not workspace_roots.temp.exists()


def test_write_partition_writes_under_the_current_primary_family_shape(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    frame = pl.DataFrame({"system_uri": ["gb:1"]})

    layer_dir = _write_partition(
        workspace_roots, system="gb", layer="cleansed", country="gb", frame=frame
    )

    assert layer_dir == layer_fixture_dir("gb", layer="cleansed")
    written = layer_dir / "primary" / "jurisdiction_code=gb" / "part-00001.parquet"
    assert written.exists()
    assert pl.read_parquet(written).equals(frame)


def test_build_sample_corpus_reads_family_split_primary_directories(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    """Real matched/ and cleansed/ output lives under a `primary/` family
    directory, not directly at the layer's own top level --
    build_sample_corpus must resolve that shape rather than only the
    pre-split layout.
    """
    matched_dir = layer_partition_dir(
        layer_fixture_dir("gleif", layer="matched"), value="gb"
    )
    matched_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1"],
            "name": ["Acme Ltd"],
            "short_name": ["acme"],
            "jurisdiction_code": ["gb"],
            "match_uri": ["gb:1"],
        }
    ).write_parquet(matched_dir / "part-00001.parquet")

    cleansed_dir = layer_partition_dir(
        layer_fixture_dir("gb", layer="cleansed"), value="gb"
    )
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb:1"],
            "name": ["Acme Ltd"],
            "jurisdiction_code": ["gb"],
        }
    ).write_parquet(cleansed_dir / "part-00001.parquet")

    sampled_source, target_frame = build_sample_corpus(
        roots=workspace_roots, source_sample=10, distractor_sample=10, seed=42
    )

    assert sampled_source.get_column("system_uri").to_list() == ["gleif:1"]
    assert target_frame.get_column("system_uri").to_list() == ["gb:1"]
