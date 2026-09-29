import json
from pathlib import Path

import polars as pl
import pytest

from blocking.comparison import build_strategy_comparison
from blocking.contracts import (
    BlockingRunConfig,
    BlockingStrategyConfig,
    blocking_run_settings,
)
from blocking.loader import load_dataset_descriptor
from blocking.reporting import produce_blocking_run, write_blocking_report
from blocking.run_layout import resolve_run_location_for
from blocking.stored_runs import (
    UnreadableRunError,
    blocking_run_config_from_record,
    load_stored_strategy_entries,
    load_strategy_run_entry,
    stored_pairings,
)
from blocking.workflow import execute_blocking_run, resolve_blocking_run_keys
from workspace.layer_layout import layer_partition_dir
from workspace.records import RECORD_FILENAME, read_record
from workspace.reference import reference_at
from workspace.roots import WorkspaceRoots


def _write_partition(
    layer_dir_for,
    *,
    system: str,
    layer: str,
    country: str,
    rows: list[dict[str, object]],
) -> Path:
    layer_dir = layer_dir_for(system, layer=layer)
    if layer == "canonical":
        # A target is read from a dated snapshot.
        layer_dir = layer_dir / "2026-01-01"
    partition_dir = layer_partition_dir(layer_dir, value=country)
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(partition_dir / "part-00001.parquet")
    return layer_dir


def _write_gleif_gb_fixture(layer_dir_for) -> None:
    matched_dir = _write_partition(
        layer_dir_for,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "acme limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
            {
                "system_uri": "gleif:2",
                "name": "beta holdings",
                "jurisdiction_code": "gb",
                "match_uri": None,
            },
        ],
    )
    (matched_dir / "_match_metadata.json").write_text(
        json.dumps({"source_system": "gleif", "target_systems": ["gb"]}) + "\n",
        encoding="utf-8",
    )
    _write_partition(
        layer_dir_for,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"},
            {"system_uri": "gb:2", "name": "omega plc", "jurisdiction_code": "gb"},
        ],
    )


def _build_config(roots: WorkspaceRoots, *, top_k: int = 2) -> BlockingRunConfig:
    return BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=load_dataset_descriptor(roots=roots, system="gleif"),
        target=load_dataset_descriptor(
            roots=roots, system="gb", require_ground_truth=False
        ),
        countries=None,
        strategy=BlockingStrategyConfig(
            representation="tfidf",
            top_k=top_k,
            min_similarity=0.2,
            max_candidates_per_source=None,
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
        ),
    )


def _produce_run(config: BlockingRunConfig) -> Path:
    keys = resolve_blocking_run_keys(config)
    with produce_blocking_run(config, keys=keys, invocation=["run"]) as staged:
        write_blocking_report(staged, execute_blocking_run(config, expected_keys=keys))
    return resolve_run_location_for(config, keys=keys).directory


def test_a_finished_run_gives_back_the_configuration_it_recorded(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    run_dir = _produce_run(config)

    record = read_record(workspace_roots, reference_at(workspace_roots, run_dir))
    assert record is not None
    rebuilt = blocking_run_config_from_record(workspace_roots, record)

    assert rebuilt.strategy == config.strategy
    assert blocking_run_settings(rebuilt) == blocking_run_settings(config)


def test_every_finished_run_of_a_pairing_becomes_a_comparison_entry(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    first = _produce_run(_build_config(workspace_roots))
    second = _produce_run(_build_config(workspace_roots, top_k=1))

    entries, unreadable = load_stored_strategy_entries(
        workspace_roots, source_system="gleif", target_system="gb"
    )

    assert unreadable == []
    assert {entry.label for entry in entries} == {
        f"tfidf/{first.name}",
        f"tfidf/{second.name}",
    }
    assert {entry.config.strategy.top_k for entry in entries} == {1, 2}
    assert all(
        entry.runtime_seconds is not None and entry.runtime_seconds >= 0
        for entry in entries
    )
    comparison = build_strategy_comparison(entries)
    assert set(comparison["label"].to_list()) == {entry.label for entry in entries}
    assert stored_pairings(workspace_roots) == [("gleif", "gb")]


def test_a_record_naming_a_setting_the_configuration_lacks_is_refused_by_name(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    run_dir = _produce_run(config)
    record_path = run_dir / RECORD_FILENAME
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    raw["parameters"]["strategy.retired_setting"] = True
    record_path.write_text(json.dumps(raw), encoding="utf-8")

    record = read_record(workspace_roots, reference_at(workspace_roots, run_dir))
    assert record is not None
    with pytest.raises(UnreadableRunError, match="retired_setting"):
        blocking_run_config_from_record(workspace_roots, record)

    entries, unreadable = load_stored_strategy_entries(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    assert entries == []
    assert len(unreadable) == 1


def test_a_directory_with_no_record_is_not_a_run(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    keys = resolve_blocking_run_keys(config)
    location = resolve_run_location_for(config, keys=keys)
    location.directory.mkdir(parents=True)

    with pytest.raises(UnreadableRunError, match="no finished run"):
        load_strategy_run_entry(workspace_roots, location)
    assert stored_pairings(workspace_roots) == []
