import json
import os
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from blocking.comparison import diff_pair_recovery
from blocking.contracts import (
    BLOCKING_ARTIFACT_SCHEMAS,
    BlockingRunConfig,
    BlockingStrategyConfig,
    blocking_run_settings,
    validate_blocking_artifact_schema,
)
from blocking.loader import load_dataset_descriptor
from blocking.reporting import (
    _difficulty_levels,
    blocking_run_inputs,
    build_blocking_summary,
    build_country_blocks,
    flatten_country_block,
    produce_blocking_run,
    write_blocking_report,
    write_pair_outcomes_report,
    write_pair_recovery_report,
    write_run_manifest,
)
from blocking.run_layout import (
    RunArtefact,
    is_finished_run,
    iter_blocking_runs,
    resolve_run_location_for,
    run_key,
)
from blocking.workflow import execute_blocking_run, resolve_blocking_run_keys
from validation.runner import NAME_EQUALITY_LEVELS, compute_pair_truth_eval
from workspace.identity import ConsumedReference, RunKeys, digest_directory
from workspace.kind_layout import Kind
from workspace.layer_layout import layer_partition_dir
from workspace.records import consumers_of, open_catalog, read_record
from workspace.reference import Side, locate, parse_reference, reference, reference_at
from workspace.roots import ENV_OUTPUT_DIR, WorkspaceRoots, resolve_workspace_roots


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
        # A target, or a source with no truth, is read from a dated snapshot.
        layer_dir = layer_dir / "2026-01-01"
    partition_dir = layer_partition_dir(layer_dir, value=country)
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(partition_dir / "part-00001.parquet")
    return layer_dir


def _write_match_metadata(
    layer_dir: Path, *, source_system: str, target_systems: list[str]
) -> None:
    metadata = {"source_system": source_system, "target_systems": target_systems}
    (layer_dir / "_match_metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )


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
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
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


def _write_gleif_gb_fixture_with_name_cleansed(layer_dir_for) -> None:
    matched_dir = _write_partition(
        layer_dir_for,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "acme limited",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_dir_for,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {
                "system_uri": "gb:1",
                "name": "acme ltd",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
        ],
    )


def _write_gleif_gb_fixture_without_ground_truth(layer_dir_for) -> None:
    _write_partition(
        layer_dir_for,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gleif:1", "name": "acme limited", "jurisdiction_code": "gb"}
        ],
    )
    _write_partition(
        layer_dir_for,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )


def _build_config(
    roots: WorkspaceRoots, *, require_source_ground_truth: bool = True
) -> BlockingRunConfig:
    source = load_dataset_descriptor(
        roots=roots, system="gleif", require_ground_truth=require_source_ground_truth
    )
    target = load_dataset_descriptor(
        roots=roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )
    return BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
    )


def test_build_blocking_summary_has_expected_shape(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))

    summary = build_blocking_summary(result)

    assert summary["candidate_pair_count"] == result.candidate_pair_count
    assert summary["raw_candidate_pair_count"] == result.raw_matched_edges.height
    assert summary["evaluation_skipped"] is False
    assert isinstance(summary["cluster_shape"], dict)
    assert isinstance(summary["directional_coverage"], dict)
    assert isinstance(summary["pruning_summary"], list)
    assert isinstance(summary["exact_match_summary"], list)
    timings = summary["timings"]
    assert isinstance(timings, list)
    assert [record["routine"] for record in timings] == [
        record["routine"] for record in result.timings
    ]
    assert timings[0]["elapsed_seconds"] >= 0.0

    # One block per scored country: the universe, then the cleansing levels in
    # cascade order, every one in the same shape so they compare like with like.
    countries = summary["countries"]
    assert isinstance(countries, list)
    block = countries[0]
    assert block["country"] == "gb"
    whole = block["whole_population"]
    assert whole["population"] == "universe"
    # Ground truth lists matching pairs only, so there is no tn to count.
    assert set(whole["confusion"]) == {"tp", "fp", "fn"}

    levels = block["name_equality_levels"]
    assert [level["population"] for level in levels] == list(NAME_EQUALITY_LEVELS)
    # A level is a population of sources, so it carries the whole matrix and
    # precision, in exactly the universe's shape.
    for level in levels:
        assert set(whole) <= set(level)
        assert set(level["confusion"]) == {"tp", "fp", "fn"}
    # The run's four keys and the systems it scored.
    assert result.keys is not None
    assert summary["keys"] == result.keys.as_identity()
    assert (summary["source_system"], summary["target_system"]) == ("gleif", "gb")


def _level_row(population: str, truth_pairs: int | None) -> dict[str, object]:
    return {"population": population, "truth_pairs": truth_pairs}


def test_difficulty_levels_report_the_drop_off_between_levels() -> None:
    """The cascade's point: each level says how many pairs were still
    unresolved when it ran and how many it took out, so the population fall
    and the score fall can be read together."""
    level_rows = {
        name: _level_row(name, count)
        for name, count in zip(
            NAME_EQUALITY_LEVELS, (12237, 4824, 1002, 300, 742), strict=True
        )
    }

    levels = _difficulty_levels(level_rows, 19105)

    assert [(level["removed_here"], level["unresolved_after"]) for level in levels] == [
        (12237, 6868),
        (4824, 2044),
        (1002, 1042),
        (300, 742),
        (742, 0),
    ]
    assert levels[0]["unresolved_before"] == 19105


def test_country_blocks_list_no_levels_for_a_run_given_no_name_forms() -> None:
    """A run whose inputs carried no name forms has no levels to classify, so
    its block lists none rather than a cascade of levels that removed nothing,
    and flattening it back gives the universe row alone."""
    pair_truth_eval = compute_pair_truth_eval(
        source_truth_map=pl.DataFrame(
            {"source_id": ["s1"], "source_match_uri": ["t1"]}
        ),
        matched_edges=pl.DataFrame({"source_id": ["s1"], "target_id": ["t1"]}),
        source_system="gleif",
        target_system="gb",
        country="gb",
        target_rows=1,
    )

    blocks = build_country_blocks(pair_truth_eval)
    assert blocks is not None
    (block,) = blocks

    assert block["name_equality_levels"] == []
    assert "unknown" not in block
    assert [row["population"] for row in flatten_country_block(block)] == ["universe"]


def test_country_blocks_round_trip_every_population() -> None:
    """Nesting a country's rows into a block and flattening it back returns
    the same populations with the same figures, so a reused run rehydrates
    exactly what its first run computed."""
    pair_truth_eval = compute_pair_truth_eval(
        source_truth_map=pl.DataFrame(
            {"source_id": ["s1", "s2"], "source_match_uri": ["t1", "t2"]}
        ),
        matched_edges=pl.DataFrame(
            {"source_id": ["s1", "s2"], "target_id": ["t1", "tX"]}
        ),
        source_system="gleif",
        target_system="gb",
        country="gb",
        target_rows=3,
        source_name_forms=pl.DataFrame(
            {
                "source_id": ["s1", "s2"],
                "name": ["acme limited", "beta holdings"],
                "name_cleansed": ["acme ltd", "beta holdings"],
            }
        ),
        target_name_forms=pl.DataFrame(
            {
                "target_id": ["t1", "t2", "tX"],
                "name": ["acme limited", "beta group", "other"],
                "name_cleansed": ["acme ltd", "beta group", "other"],
            }
        ),
    )

    blocks = build_country_blocks(pair_truth_eval)
    assert blocks is not None
    (block,) = blocks
    rehydrated = pl.DataFrame(
        [
            {"source_system": "gleif", "target_system": "gb", **row}
            for row in flatten_country_block(block)
        ],
        schema=pair_truth_eval.schema,
    )

    assert rehydrated.equals(pair_truth_eval)


def test_build_blocking_summary_without_ground_truth(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture_without_ground_truth(layer_fixture_dir)
    config = _build_config(workspace_roots, require_source_ground_truth=False)

    result = execute_blocking_run(config)
    summary = build_blocking_summary(result)

    assert summary["countries"] is None
    assert summary["raw_countries"] is None
    # A run made with --ground-truth false says so in its own
    # summary, rather than that being merely implied by countries being None.
    assert summary["evaluation_skipped"] is True


def test_write_blocking_report_writes_expected_files(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))
    output_dir = tmp_path / "report_out"

    written = write_blocking_report(run_location(output_dir), result)

    # Row-level data as parquet, everything summary-shaped in summary.json.
    # A one-row parquet per summary is what made this directory a dozen files
    # nobody could tell apart. The per-pair detail is row-level too, and is
    # present on every run with ground truth now that a run derives its own
    # name forms rather than depending on the layer carrying them.
    written_names = {path.name for path in written}
    assert written_names == {
        "matched_edges.parquet",
        "raw_matched_edges.parquet",
        "clusters.parquet",
        # Written unconditionally, empty here since this config
        # leaves target-neighbour linking off.
        "clusters_before_union.parquet",
        "target_neighbor_edges.parquet",
        "pair_truth_eval_detail.parquet",
        "summary.json",
    }
    for path in written:
        assert path.exists()

    reread_edges = pl.read_parquet(output_dir / "matched_edges.parquet")
    assert reread_edges.columns == result.matched_edges.columns

    summary_payload = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["candidate_pair_count"] == result.candidate_pair_count


def test_write_blocking_report_with_diagnostics_writes_extra_files(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = replace(_build_config(workspace_roots), emit_diagnostics=True)
    result = execute_blocking_run(config)
    output_dir = tmp_path / "report_out"

    written = write_blocking_report(run_location(output_dir), result)

    written_names = {path.name for path in written}
    assert "source_diagnostics.parquet" in written_names
    assert "target_diagnostics.parquet" in written_names

    reread_source = pl.read_parquet(output_dir / "source_diagnostics.parquet")
    assert result.source_diagnostics is not None
    assert reread_source.columns == result.source_diagnostics.columns


def test_write_blocking_report_writes_pair_truth_eval_detail_when_computable(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    _write_gleif_gb_fixture_with_name_cleansed(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))
    output_dir = tmp_path / "report_out"

    written = write_blocking_report(run_location(output_dir), result)

    written_names = {path.name for path in written}
    assert "pair_truth_eval_detail.parquet" in written_names

    assert result.pair_truth_eval_detail is not None
    reread = pl.read_parquet(output_dir / "pair_truth_eval_detail.parquet")
    assert reread.columns == result.pair_truth_eval_detail.columns
    validate_blocking_artifact_schema(reread, artifact_name="pair_truth_eval_detail")


def test_write_blocking_report_without_diagnostics_omits_diagnostic_files(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))
    output_dir = tmp_path / "report_out"

    written = write_blocking_report(run_location(output_dir), result)

    written_names = {path.name for path in written}
    assert "source_diagnostics.parquet" not in written_names
    assert "target_diagnostics.parquet" not in written_names


def test_write_blocking_report_without_ground_truth_omits_truth_eval_files(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    _write_gleif_gb_fixture_without_ground_truth(layer_fixture_dir)
    config = _build_config(workspace_roots, require_source_ground_truth=False)
    result = execute_blocking_run(config)
    output_dir = tmp_path / "report_out"

    written = write_blocking_report(run_location(output_dir), result)

    written_names = {path.name for path in written}
    assert "pair_truth_eval.parquet" not in written_names
    assert "raw_pair_truth_eval.parquet" not in written_names


def _write_gleif_gb_fixture_with_near_duplicate_targets(layer_dir_for) -> None:
    """`gb:1`/`gb:2` are near-duplicate target names, near enough for the
    target-neighbour probe to link them at the threshold
    `_build_config_with_target_neighbors` uses; `gb:3` shares nothing with
    either and stays unlinked, the negative control."""
    matched_dir = _write_partition(
        layer_dir_for,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "acme trading ltd",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_dir_for,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {
                "system_uri": "gb:1",
                "name": "acme trading ltd",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:2",
                "name": "acme trading group ltd",
                "jurisdiction_code": "gb",
            },
            {"system_uri": "gb:3", "name": "omega plc", "jurisdiction_code": "gb"},
        ],
    )


def _build_config_with_target_neighbors(roots: WorkspaceRoots) -> BlockingRunConfig:
    config = _build_config(roots)
    return replace(
        config,
        strategy=replace(
            config.strategy,
            # Verified directly against this fixture's own raw cosine
            # similarity: gb:1/gb:2 (the near-duplicate pair) score 0.90,
            # gb:2/gb:3 (the unrelated target, sharing only generic
            # single-character n-grams under this config's tfidf_ngram_min=1)
            # scores 0.50 -- 0.7 clears the former and excludes the latter.
            target_neighbor_min_similarity=0.7,
            target_neighbor_max_per_target=5,
        ),
    )


def test_build_blocking_summary_reports_target_neighbor_linking(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture_with_near_duplicate_targets(layer_fixture_dir)
    result = execute_blocking_run(_build_config_with_target_neighbors(workspace_roots))

    # gb:1/gb:2 are near-duplicate names, linked by the probe; gb:3 shares
    # nothing with either and never features in a target-neighbour edge.
    assert result.target_neighbor_edges.height > 0
    linked_ids = set(
        result.target_neighbor_edges.get_column("target_id_a").to_list()
    ) | set(result.target_neighbor_edges.get_column("target_id_b").to_list())
    assert linked_ids == {"gb:1", "gb:2"}
    # The union grows gb:1's cluster to include gb:2, which no source ever
    # scored a candidate edge to -- the whole point of the item.
    before_cluster = dict(
        zip(
            result.clusters_before_union.get_column("node_id").to_list(),
            result.clusters_before_union.get_column("cluster_id").to_list(),
            strict=True,
        )
    )
    after_cluster = dict(
        zip(
            result.clusters.get_column("node_id").to_list(),
            result.clusters.get_column("cluster_id").to_list(),
            strict=True,
        )
    )
    assert "gb:2" not in before_cluster
    assert after_cluster.get("gb:2") == after_cluster.get("gb:1")

    summary = build_blocking_summary(result)

    # gleif:1 (the source), gb:1 (its own match) and gb:2 (the near-duplicate
    # the union adds) -- three nodes after the union, two (source + gb:1)
    # before it.
    cluster_shape = summary["cluster_shape"]
    cluster_shape_before_union = summary["cluster_shape_before_union"]
    assert isinstance(cluster_shape, dict)
    assert isinstance(cluster_shape_before_union, dict)
    assert cluster_shape["max_cluster_size"] == 3
    assert cluster_shape_before_union["max_cluster_size"] == 2
    neighbor_summary = summary["target_neighbor_summary"]
    assert isinstance(neighbor_summary, list)
    assert neighbor_summary[0]["min_similarity"] == 0.7
    assert neighbor_summary[0]["max_per_target"] == 5
    assert neighbor_summary[0]["edge_count"] > 0


def test_write_blocking_report_writes_target_neighbor_files(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    _write_gleif_gb_fixture_with_near_duplicate_targets(layer_fixture_dir)
    result = execute_blocking_run(_build_config_with_target_neighbors(workspace_roots))
    output_dir = tmp_path / "report_out"

    written = write_blocking_report(run_location(output_dir), result)

    written_names = {path.name for path in written}
    assert "clusters_before_union.parquet" in written_names
    assert "target_neighbor_edges.parquet" in written_names
    reread = pl.read_parquet(output_dir / "target_neighbor_edges.parquet")
    assert reread.height == result.target_neighbor_edges.height
    validate_blocking_artifact_schema(reread, artifact_name="target_neighbor_edges")


def test_validate_blocking_artifact_schema_rejects_missing_column() -> None:
    frame = pl.DataFrame({"source_id": ["s1"], "target_id": ["t1"]})

    with pytest.raises(ValueError, match="matched_edges is missing required columns"):
        validate_blocking_artifact_schema(frame, artifact_name="matched_edges")


def test_validate_blocking_artifact_schema_accepts_valid_frame() -> None:
    frame = pl.DataFrame(
        {
            "source_id": ["s1"],
            "target_id": ["t1"],
            "similarity": [0.9],
            "rank": [1],
            "country": ["gb"],
        }
    )

    validate_blocking_artifact_schema(frame, artifact_name="matched_edges")


def test_validate_blocking_artifact_schema_rejects_pair_truth_eval_detail_missing_column() -> (
    None
):
    frame = pl.DataFrame({"source_id": ["s1"], "target_id": ["t1"]})

    with pytest.raises(
        ValueError, match="pair_truth_eval_detail is missing required columns"
    ):
        validate_blocking_artifact_schema(frame, artifact_name="pair_truth_eval_detail")


def test_validate_blocking_artifact_schema_accepts_valid_pair_truth_eval_detail_frame() -> (
    None
):
    frame = pl.DataFrame(
        {
            "source_system": ["gleif"],
            "target_system": ["gb"],
            "country": ["gb"],
            "source_id": ["s1"],
            "target_id": ["t1"],
            "source_name": ["Acme Ltd"],
            "source_name_cleansed": ["acme ltd"],
            "target_name": ["Acme Ltd"],
            "target_name_cleansed": ["acme ltd"],
            "name_equality": ["raw"],
            "is_truth_pair": [True],
            "found": [True],
            "similarity": [1.0],
            "rank": [1],
        }
    )

    validate_blocking_artifact_schema(frame, artifact_name="pair_truth_eval_detail")


def test_validate_blocking_artifact_schema_rejects_unknown_artifact_name() -> None:
    frame = pl.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="Unknown artifact_name"):
        validate_blocking_artifact_schema(frame, artifact_name="not_a_real_artifact")


def test_write_run_manifest_records_identity_configuration_and_commit(
    tmp_path: Path, workspace_roots: WorkspaceRoots, run_location
) -> None:
    identity = {"strategy.top_k": 5, "countries": ["gb"], "root_like": Path("x")}
    configuration = {"top_k": {"value": 5, "source": "given", "surface": "--top-k"}}

    path = write_run_manifest(
        run_location(tmp_path / "run"),
        identity=identity,
        configuration=configuration,
        roots=workspace_roots,
        commit="abc1234",
    )

    assert path == tmp_path / "run" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["identity"]["strategy.top_k"] == 5
    assert manifest["identity"]["root_like"] == "x"
    assert manifest["configuration"] == configuration
    assert manifest["commit"] == "abc1234"


def test_write_run_manifest_records_every_root_with_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_location
) -> None:
    """A run says where it read from and wrote to: all five roots, each with
    the source its value came from, so a relocated run is not invisible."""
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path / "elsewhere"))
    roots = resolve_workspace_roots(
        tmp_path, data_dir=tmp_path / "given-data", environ=os.environ
    )

    path = write_run_manifest(
        run_location(tmp_path / "run"),
        identity={},
        configuration={},
        roots=roots,
        commit=None,
    )

    recorded = json.loads(path.read_text(encoding="utf-8"))["workspace_roots"]
    assert set(recorded) == {"checkout", "config", "data", "artifacts", "temp"}
    assert recorded["data"] == {
        "path": str((tmp_path / "given-data").resolve()),
        "source": "flag",
    }
    assert recorded["artifacts"]["source"] == "environment"
    assert recorded["temp"]["source"] == "default"
    assert recorded["checkout"]["source"] == "checkout"


def test_write_pair_recovery_report_writes_parquet(
    tmp_path: Path, comparison_location
) -> None:
    baseline = pl.DataFrame(
        {
            "source_system": ["gleif"],
            "target_system": ["gb"],
            "country": ["gb"],
            "source_id": ["gleif:1"],
            "target_id": ["gb:1"],
            "source_name": ["gleif:1"],
            "source_name_cleansed": ["gleif:1"],
            "target_name": ["gb:1"],
            "target_name_cleansed": ["gb:1"],
            "name_equality": ["never"],
            "is_truth_pair": [True],
            "found": [False],
            "similarity": [None],
            "rank": [None],
        }
    )
    variant = baseline.with_columns(
        pl.lit(True).alias("found"),
        pl.lit(0.5).alias("similarity"),
        pl.lit(1).alias("rank"),
    )

    recovery = diff_pair_recovery(baseline, variant)
    output_dir = tmp_path / "recovery_out"
    path = write_pair_recovery_report(comparison_location(output_dir), recovery)

    assert path == output_dir / "pair_recovery_attribution.parquet"
    assert path.exists()
    reread = pl.read_parquet(path)
    assert reread.height == 1
    assert reread.get_column("source_id").to_list() == ["gleif:1"]


def _pair_outcome_frames(
    *, outcome_label: str, candidate_label: str = "tfidf/a"
) -> dict[str, pl.DataFrame]:
    """A runs table holding `tfidf/a`, one outcome row naming `outcome_label`
    and one candidate row, overall and per source, one recall-curve row and
    one audit row, each naming `candidate_label`."""
    runs_schema = BLOCKING_ARTIFACT_SCHEMAS["pair_outcome_runs"]
    outcomes_schema = BLOCKING_ARTIFACT_SCHEMAS["pair_outcomes"]
    candidates = pl.DataFrame(
        [
            {
                "label": candidate_label,
                "min_similarity": 0.75,
                "rank": 1,
                "candidates": 1,
            }
        ],
        schema=BLOCKING_ARTIFACT_SCHEMAS["pair_outcome_candidates"],
    )
    source_candidates = pl.DataFrame(
        [
            {
                "label": candidate_label,
                "country": "gb",
                "source_id": "gleif:1",
                "min_similarity": 0.75,
                "rank": 1,
                "candidates": 1,
            }
        ],
        schema=BLOCKING_ARTIFACT_SCHEMAS["pair_outcome_source_candidates"],
    )
    runs = pl.DataFrame(
        [{**dict.fromkeys(runs_schema, None), "label": "tfidf/a"}], schema=runs_schema
    )
    outcomes = pl.DataFrame(
        [
            {
                **dict.fromkeys(outcomes_schema, None),
                "label": outcome_label,
                "source_id": "gleif:1",
                "found": True,
            }
        ],
        schema=outcomes_schema,
    )
    curves_schema = BLOCKING_ARTIFACT_SCHEMAS["pair_outcome_recall_curves"]
    recall_curves = pl.DataFrame(
        [{**dict.fromkeys(curves_schema, None), "label": candidate_label}],
        schema=curves_schema,
    )
    audits_schema = BLOCKING_ARTIFACT_SCHEMAS["pair_audit"]
    audits = pl.DataFrame(
        [{**dict.fromkeys(audits_schema, None), "label": candidate_label}],
        schema=audits_schema,
    )
    return {
        "runs": runs,
        "outcomes": outcomes,
        "candidates": candidates,
        "source_candidates": source_candidates,
        "recall_curves": recall_curves,
        "audits": audits,
    }


def test_write_pair_outcomes_report_writes_the_runs_and_their_outcomes(
    tmp_path: Path, comparison_location
) -> None:
    frames = _pair_outcome_frames(outcome_label="tfidf/a")
    output_dir = tmp_path / "outcomes_out"

    paths = write_pair_outcomes_report(comparison_location(output_dir), **frames)

    assert paths == [
        output_dir / "pair_outcome_runs.parquet",
        output_dir / "pair_outcomes.parquet",
        output_dir / "pair_outcome_candidates.parquet",
        output_dir / "pair_outcome_source_candidates.parquet",
        output_dir / "pair_outcome_recall_curves.parquet",
        output_dir / "pair_outcome_audits.parquet",
    ]
    assert pl.read_parquet(paths[2]).get_column("candidates").to_list() == [1]
    assert pl.read_parquet(paths[3]).get_column("source_id").to_list() == ["gleif:1"]
    assert pl.read_parquet(paths[0]).get_column("label").to_list() == ["tfidf/a"]
    assert pl.read_parquet(paths[1]).get_column("source_id").to_list() == ["gleif:1"]


def test_write_pair_outcomes_report_refuses_an_outcome_of_an_unlisted_run(
    tmp_path: Path, comparison_location
) -> None:
    output_dir = tmp_path / "outcomes_out"

    with pytest.raises(ValueError, match="tfidf/b"):
        write_pair_outcomes_report(
            comparison_location(output_dir),
            **_pair_outcome_frames(outcome_label="tfidf/b"),
        )
    with pytest.raises(ValueError, match="tfidf/c"):
        write_pair_outcomes_report(
            comparison_location(output_dir),
            **_pair_outcome_frames(outcome_label="tfidf/a", candidate_label="tfidf/c"),
        )

    assert not output_dir.exists()


# -- the production record ---------------------------------------------------------


def _produce_run(roots: WorkspaceRoots) -> tuple[BlockingRunConfig, RunKeys]:
    config = _build_config(roots)
    keys = resolve_blocking_run_keys(config)
    with produce_blocking_run(config, keys=keys, invocation=["run"]) as staged:
        write_blocking_report(staged, execute_blocking_run(config, expected_keys=keys))
    return config, keys


def test_a_run_leaves_a_record_naming_both_layers_under_its_folder_key(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)

    config, keys = _produce_run(workspace_roots)

    location = resolve_run_location_for(config, keys=keys)
    record = read_record(
        workspace_roots, reference_at(workspace_roots, location.directory)
    )
    assert record is not None
    assert record.key == location.directory.name == run_key(keys)
    assert {
        role: (used.uri, used.content_digest) for role, used in record.inputs.items()
    } == {
        "source": ("data://gleif/matched", keys.source_population),
        "target": ("data://gb/canonical/2026-01-01", keys.target_population),
    }
    assert record.parameters == json.loads(
        json.dumps(blocking_run_settings(config), default=str)
    )
    assert location.path(RunArtefact.SUMMARY).is_file()
    assert [run.directory for run in iter_blocking_runs(workspace_roots)] == [
        location.directory
    ]


def test_an_interrupted_run_leaves_no_location_and_no_record(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    keys = resolve_blocking_run_keys(config)

    with (
        pytest.raises(KeyboardInterrupt),
        produce_blocking_run(config, keys=keys, invocation=["run"]) as staged,
    ):
        write_blocking_report(staged, execute_blocking_run(config))
        raise KeyboardInterrupt

    location = resolve_run_location_for(config, keys=keys)
    assert not location.directory.exists()
    assert list(location.directory.parent.iterdir()) == []
    assert list(iter_blocking_runs(workspace_roots)) == []
    assert not is_finished_run(workspace_roots, location)


def test_the_catalog_lists_every_run_that_consumed_a_layer(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config, keys = _produce_run(workspace_roots)
    run_uri = reference_at(
        workspace_roots, resolve_run_location_for(config, keys=keys).directory
    ).uri

    catalog = open_catalog(workspace_roots)

    for layer in (
        "data://gleif/matched",
        "data://gb/canonical/2026-01-01",
    ):
        assert consumers_of(catalog, parse_reference(layer)) == [run_uri]
    assert consumers_of(catalog, parse_reference("data://ie/matched")) == []


def test_a_perturbed_source_is_recorded_as_its_dataset_and_its_profile(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    perturbed = replace(
        config, source=replace(config.source, system="perturbed://gb/en-lite/v2/42")
    )
    keys = resolve_blocking_run_keys(config)
    promoted = reference(Kind.PERTURBATION, Side.PROFILE, name="en-lite", version="v2")
    draft = reference(
        Kind.PERTURBATION, Side.PROFILE, name="en-lite", version="v2", stage="draft"
    )

    draft_dir = locate(workspace_roots, draft)
    draft_dir.mkdir(parents=True)
    (draft_dir / "profile.json").write_text("{}", encoding="utf-8")
    inputs = blocking_run_inputs(perturbed, keys=keys)
    assert inputs["source"] == ConsumedReference(
        parse_reference("perturbed://gb/en-lite/v2/42"), keys.source_population
    )
    assert inputs["source_profile"] == ConsumedReference(
        draft, digest_directory(draft_dir)
    )

    locate(workspace_roots, promoted).mkdir(parents=True)
    assert blocking_run_inputs(perturbed, keys=keys)["source_profile"] == (
        ConsumedReference(promoted)
    )
