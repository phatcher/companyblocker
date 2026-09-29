from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest
from company_vectorize.comparison_protocol import (
    FIXED_SLICES,
    MANDATORY_SUITE,
)
from company_vectorize.dense_vocabulary_gate import (
    DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
)
from company_vectorize.partition_similarity import DEFAULT_KMEANS_START

from blocking.contracts import blocking_run_settings
from blocking.loader import load_dataset_descriptor
from blocking.run_layout import (
    ComparisonArtefact,
    resolve_comparison_location,
    resolve_pairing_dir,
    resolve_run_location,
)
from scripts import compare_blocking_strategies, run_blocking
from workspace.artifact_layout import blocking_artifact_root
from workspace.identity import RunKeys
from workspace.layer_layout import layer_partition_dir
from workspace.roots import (
    ENV_DATA_DIR,
    ENV_OUTPUT_DIR,
    ENV_TEMP_DIR,
    ENVIRONMENT_VARIABLES,
    WorkspaceRoots,
    default_workspace_roots,
)


def test_build_parser_similarity_backend_accepts_partition_backends() -> None:
    # --similarity-backend's choices must include kmeans/hdbscan
    # (previously hardcoded to sklearn/sparse_dot_topn/svd_rerank), each
    # pairable with the new --kmeans-clusters/--min-cluster-size options.
    for backend, flag, value in (
        ("kmeans", "--kmeans-clusters", "10"),
        ("hdbscan", "--min-cluster-size", "3"),
    ):
        args = compare_blocking_strategies.build_parser().parse_args(
            [
                "--source",
                "gleif",
                "--target",
                "gb",
                "--similarity-backend",
                backend,
                flag,
                value,
            ]
        )
        assert args.similarity_backend == backend


def test_partition_backend_options_left_unset_stay_off_the_cell_argv() -> None:
    # An option never set is not passed to the cell's run_blocking.py, which
    # resolves the backend's own default into the run's identity; the
    # comparison's own view of the cell names the same settings.
    for backend in ("kmeans", "hdbscan"):
        args = compare_blocking_strategies.build_parser().parse_args(
            [
                "--source",
                "gleif",
                "--target",
                "gb",
                "--similarity-backend",
                backend,
            ]
        )
        strategy = compare_blocking_strategies._resolve_strategy_config(
            "tfidf", backend, args
        )
        argv = compare_blocking_strategies._build_cell_argv(
            representation="tfidf",
            similarity_backend=backend,
            source_system="gleif",
            target_system="gb",
            country="gb",
            args=args,
        )

        assert "backend.kmeans_clusters" not in " ".join(argv)
        assert "backend.min_cluster_size" not in " ".join(argv)
        cell = run_blocking._resolve_strategy(
            run_blocking.build_parser().parse_args(argv[2:])
        )
        assert strategy.backend_options == cell.backend_options


def test_a_given_partition_backend_option_reaches_the_strategy() -> None:
    args = compare_blocking_strategies.build_parser().parse_args(
        ["--similarity-backend", "kmeans", "--kmeans-clusters", "50"]
    )

    strategy = compare_blocking_strategies._resolve_strategy_config(
        "tfidf", "kmeans", args
    )

    assert strategy.backend_options is not None
    assert strategy.backend_options["kmeans_clusters"] == 50
    assert strategy.backend_options["kmeans_start"] == DEFAULT_KMEANS_START


def test_build_parser_gate_flags_default_to_the_documented_threshold() -> None:
    args = compare_blocking_strategies.build_parser().parse_args(
        ["--source", "gleif", "--target", "gb"]
    )
    assert args.max_rows == DEFAULT_DENSE_VOCABULARY_MAX_ROWS
    assert args.force is False


def test_build_parser_source_and_target_system_default_to_none() -> None:
    # Omitting both is what triggers the fixed-protocol sweep.
    args = compare_blocking_strategies.build_parser().parse_args([])
    assert args.source_system is None
    assert args.target_system is None


def test_build_parser_runtime_budget_defaults_to_the_protocol_constant() -> None:
    from company_vectorize.comparison_protocol import RUNTIME_BUDGET_SECONDS

    args = compare_blocking_strategies.build_parser().parse_args([])
    assert args.runtime_budget_seconds == RUNTIME_BUDGET_SECONDS


def test_build_parser_representations_default_includes_sbert() -> None:
    # sbert is part of MANDATORY_SUITE now that its optional
    # sentence-transformers dependency is declared and synced -- it is no
    # longer held out of the default sweep the way BASELINE_SUITE held it
    # out.
    args = compare_blocking_strategies.build_parser().parse_args([])
    assert "sbert" in args.representations
    assert set(args.representations) == set(compare_blocking_strategies.REPRESENTATIONS)


def test_build_parser_representations_narrows_to_one() -> None:
    args = compare_blocking_strategies.build_parser().parse_args(
        ["--representations", "sbert"]
    )
    assert args.representations == ["sbert"]


def test_build_parser_sbert_model_defaults_to_none() -> None:
    args = compare_blocking_strategies.build_parser().parse_args([])
    assert args.sbert_model_name is None


def test_resolve_strategy_config_forwards_sbert_model_only_for_sbert() -> None:
    args = compare_blocking_strategies.build_parser().parse_args(
        ["--representations", "sbert", "--sbert-model", "fr-sentence-camembert-base"]
    )

    sbert_strategy = compare_blocking_strategies._resolve_strategy_config(
        "sbert", "hnsw", args
    )
    tfidf_strategy = compare_blocking_strategies._resolve_strategy_config(
        "tfidf", "kmeans", args
    )

    assert sbert_strategy.sbert_model_name == "fr-sentence-camembert-base"
    assert tfidf_strategy.sbert_model_name is None


def test_tokenizer_for_sbert_and_tfidf_both_fall_back_to_wordpiece() -> None:
    assert compare_blocking_strategies._tokenizer_for("tfidf") == "wordpiece"
    assert compare_blocking_strategies._tokenizer_for("sbert") == "wordpiece"
    assert compare_blocking_strategies._tokenizer_for("wordpiece") == "wordpiece"
    assert (
        compare_blocking_strategies._tokenizer_for("sentencepiece") == "sentencepiece"
    )


def test_build_cell_argv_forwards_sbert_model_only_for_sbert_and_when_set(
    tmp_path: Path,
) -> None:
    args = compare_blocking_strategies.build_parser().parse_args(
        ["--representations", "sbert", "--sbert-model", "fr-sentence-camembert-base"]
    )

    sbert_argv = compare_blocking_strategies._build_cell_argv(
        representation="sbert",
        similarity_backend="hnsw",
        source_system="gleif",
        target_system="fr",
        country="fr",
        args=args,
    )
    tfidf_argv = compare_blocking_strategies._build_cell_argv(
        representation="tfidf",
        similarity_backend="kmeans",
        source_system="gleif",
        target_system="fr",
        country="fr",
        args=args,
    )

    assert "--sbert-model" in sbert_argv
    assert sbert_argv[sbert_argv.index("--sbert-model") + 1] == (
        "fr-sentence-camembert-base"
    )
    assert "--sbert-model" not in tfidf_argv
    # No tokenizer.name=sbert additional-arg -- sbert has no tokenizer stage.
    assert "tokenizer.name=sbert" not in " ".join(sbert_argv)


def test_resolve_slices_defaults_to_the_fixed_protocol_slices() -> None:
    args = compare_blocking_strategies.build_parser().parse_args([])
    assert compare_blocking_strategies._resolve_slices(
        args, default_workspace_roots(Path.cwd())
    ) == list(FIXED_SLICES)


def test_resolve_slices_honours_an_explicit_single_pair() -> None:
    args = compare_blocking_strategies.build_parser().parse_args(
        ["--source", "gleif", "--target", "fr"]
    )
    assert compare_blocking_strategies._resolve_slices(
        args, default_workspace_roots(Path.cwd())
    ) == [("gleif", "fr")]


def test_resolve_slices_rejects_a_one_sided_override() -> None:
    args = compare_blocking_strategies.build_parser().parse_args(["--source", "gleif"])
    with pytest.raises(ValueError):
        compare_blocking_strategies._resolve_slices(
            args, default_workspace_roots(Path.cwd())
        )


@pytest.mark.integration
def test_execute_cell_subprocess_records_timeout_and_does_not_raise() -> None:
    # A fake/fast target rather than a real multi-minute run --
    # this only exercises subprocess.run(..., timeout=...) handling.
    status, elapsed, message = compare_blocking_strategies._execute_cell_subprocess(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=0.2
    )
    assert status == "timeout"
    assert message is not None
    assert elapsed < 5


def test_execute_cell_subprocess_records_error_on_nonzero_exit() -> None:
    status, _elapsed, message = compare_blocking_strategies._execute_cell_subprocess(
        [sys.executable, "-c", "import sys; sys.exit(3)"], timeout_seconds=5
    )
    assert status == "error"
    assert message is not None


def test_execute_cell_subprocess_records_completed_on_a_clean_exit() -> None:
    status, _elapsed, message = compare_blocking_strategies._execute_cell_subprocess(
        [sys.executable, "-c", "pass"], timeout_seconds=5
    )
    assert status == "completed"
    assert message is None


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


def _write_gleif_gb_fixture(layer_dir_for) -> None:
    matched_dir = _write_partition(
        layer_dir_for,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "Acme Limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            }
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
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )


def _roots_argv(roots: WorkspaceRoots) -> list[str]:
    return [
        "--data-dir",
        str(roots.data),
        "--output-dir",
        str(roots.artifacts),
        "--temp-dir",
        str(roots.temp),
    ]


def test_dry_run_reports_the_resolved_sweep_without_running_any_cell(
    workspace_roots: WorkspaceRoots,
    capsys,
) -> None:
    """--dry-run resolves the slice/representation/country plan (the
    argument-resolution this script exists to get right), names what it would
    write under --output-dir, and stops before `_run_one_cell` would spawn a
    single `run_blocking.py` subprocess -- so it pays none of the per-cell
    subprocess cost the two `integration` tests below pay to reach the same
    wiring facts as a side effect."""
    exit_code = compare_blocking_strategies.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
            "--representations",
            "tfidf",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run] compare_blocking_strategies: would proceed with:" in out
    assert "slices=[('gleif', 'gb')]" in out
    assert "representations=['tfidf']" in out
    assert "countries=['gb'] (given)" in out
    assert "prefix_filter=False (default)" in out
    assert "kmeans_clusters" not in out
    pairing_dir = resolve_pairing_dir(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    comparison = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    assert f"would extend {pairing_dir}" in out
    assert f"would clear {comparison.path(ComparisonArtefact.REPORT)}" in out
    assert f"would clear {comparison.path(ComparisonArtefact.SUITE_STATUS)}" in out
    assert "name_equality_attribution" not in out
    assert not blocking_artifact_root(workspace_roots).exists()


@pytest.mark.integration
def test_gated_reference_cells_are_recorded_as_failed_cells_not_aborted(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    # --reference-column selects REFERENCE_SUITE's sklearn leg, the
    # one the dense-vocabulary gate can refuse. A per-cell refusal is
    # recorded to the sidecar failures log and the sweep continues -- the
    # tfidf leg's own artefacts and comparison report still get written.
    # Every cell's subprocess finds the fixture only through the roots this
    # invocation hands it, so a completed cell proves they reached it.
    _write_gleif_gb_fixture(layer_fixture_dir)

    exit_code = compare_blocking_strategies.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
            "--representations",
            "tfidf",
            "wordpiece",
            "sentencepiece",
            "--reference-column",
            "--max-rows",
            "0",
        ]
    )

    assert exit_code == 0
    # A completed cell's run lands in the shared pairing tree, keyed by its own
    # identity beneath the representation -- the same place a standalone
    # `run_blocking.py` would file it, not a corner this script composed.
    pairing_dir = resolve_pairing_dir(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    tfidf_runs = list(pairing_dir.glob("tfidf/*/summary.json"))
    assert len(tfidf_runs) == 1

    comparison_location = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    failures_path = comparison_location.path(ComparisonArtefact.FAILURES)
    assert failures_path.exists()
    failure_rows = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["label"] for row in failure_rows} == {"wordpiece", "sentencepiece"}
    assert all(row["similarity_backend"] == "sklearn" for row in failure_rows)
    assert all(row["status"] == "error" for row in failure_rows)
    assert all(row["country"] == "gb" for row in failure_rows)
    assert all(row["target_system"] == "gb" for row in failure_rows)
    assert all("--force" in row["error_message"] for row in failure_rows)

    assert comparison_location.path(ComparisonArtefact.REPORT).exists()

    # tfidf/gb's reference cell completed; wordpiece/sentencepiece's were
    # gated (max-rows=0) -- reported as refused, not merely missing, and the
    # mandatory suite (never run under --reference-column) is reported
    # missing in full, independently.
    status_path = comparison_location.path(ComparisonArtefact.SUITE_STATUS)
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["complete"] is False
    assert len(status["missing_cells"]) == len(MANDATORY_SUITE)
    refused_representations = {
        cell["representation"] for cell in status["refused_reference_cells"]
    }
    assert refused_representations == {"wordpiece", "sentencepiece"}
    assert all(
        cell["similarity_backend"] == "sklearn"
        for cell in status["refused_reference_cells"]
    )
    assert {
        "representation": "tfidf",
        "similarity_backend": "sklearn",
        "country": "gb",
    } not in status["missing_reference_cells"]
    assert {
        "representation": "sbert",
        "similarity_backend": "dense_brute",
        "country": "gb",
    } in status["missing_reference_cells"]


def test_suite_status_reports_every_mandatory_cell_a_sweep_did_not_complete(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A sweep reports against the full `MANDATORY_SUITE` whatever it ran:
    a completed tfidf/kmeans cell leaves every other cell missing."""
    location = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    location.directory.mkdir(parents=True)
    comparison = pl.DataFrame(
        {
            "representation": ["tfidf"],
            "similarity_backend": ["kmeans"],
            "country": ["gb"],
            "stage": ["pruned"],
        }
    )

    compare_blocking_strategies._write_baseline_suite_status(
        comparison=comparison, countries=("gb",), location=location, failures=[]
    )

    status = json.loads(
        location.path(ComparisonArtefact.SUITE_STATUS).read_text(encoding="utf-8")
    )
    assert status["complete"] is False
    missing_cells = {
        (cell["representation"], cell["similarity_backend"])
        for cell in status["missing_cells"]
    }
    assert ("tfidf", "kmeans") not in missing_cells
    assert len(missing_cells) == len(MANDATORY_SUITE) - 1


def test_only_a_gate_refusal_of_a_dense_vocabulary_cell_reads_as_refused() -> None:
    gate_message = compare_blocking_strategies.REFUSAL_MESSAGE_MARKER
    failures: list[dict[str, object]] = [
        {
            "label": "wordpiece",
            "similarity_backend": "sklearn",
            "country": "gb",
            "status": "error",
            "error_message": f"... {gate_message} ...",
        },
        {
            "label": "wordpiece",
            "similarity_backend": "sklearn",
            "country": "ie",
            "status": "timeout",
            "error_message": gate_message,
        },
        {
            "label": "tfidf",
            "similarity_backend": "sklearn",
            "country": "gb",
            "status": "error",
            "error_message": gate_message,
        },
    ]

    assert compare_blocking_strategies._refused_reference_cells(failures) == [
        ("wordpiece", "sklearn", "gb")
    ]


def test_cell_environment_hands_a_cell_this_invocation_s_roots(
    workspace_roots: WorkspaceRoots,
) -> None:
    env = compare_blocking_strategies._cell_environment(workspace_roots)

    assert env[ENV_DATA_DIR] == str(workspace_roots.data)
    assert env[ENV_OUTPUT_DIR] == str(workspace_roots.artifacts)
    assert env[ENV_TEMP_DIR] == str(workspace_roots.temp)


def test_run_one_cell_records_an_invalid_backend_as_a_failure_not_a_crash(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A suite can name a backend `blocking.contracts` does not accept:
    config-validation's `ValueError` is caught here and turned into a failed
    cell, never left to propagate and abort the whole run of cells."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    args = compare_blocking_strategies.build_parser().parse_args(
        [
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
        ]
    )

    entry, failure = compare_blocking_strategies._run_one_cell(
        representation="tfidf",
        similarity_backend="no_such_backend",
        source_system="gleif",
        target_system="gb",
        country="gb",
        args=args,
        roots=workspace_roots,
        source_descriptor=load_dataset_descriptor(
            roots=workspace_roots, system="gleif"
        ),
        target_descriptor=load_dataset_descriptor(
            roots=workspace_roots, system="gb", require_ground_truth=False
        ),
        runtime_budget_seconds=5,
    )

    assert entry is None
    assert failure is not None
    assert failure["status"] == "error"
    assert failure["similarity_backend"] == "no_such_backend"
    assert "similarity_backend" in str(failure["error_message"])


def test_run_one_cell_completes_a_mandatory_lsh_cell(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """`MANDATORY_SUITE` names `lsh` for every classical representation, and
    blocking accepts it, so the cell is measured and not recorded as failed."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    args = compare_blocking_strategies.build_parser().parse_args(
        [
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
        ]
    )

    entry, failure = compare_blocking_strategies._run_one_cell(
        representation="tfidf",
        similarity_backend="lsh",
        source_system="gleif",
        target_system="gb",
        country="gb",
        args=args,
        roots=workspace_roots,
        source_descriptor=load_dataset_descriptor(
            roots=workspace_roots, system="gleif"
        ),
        target_descriptor=load_dataset_descriptor(
            roots=workspace_roots, system="gb", require_ground_truth=False
        ),
        runtime_budget_seconds=120,
    )

    assert failure is None
    assert entry is not None


def _gleif_gb_args(*extra: str):
    return compare_blocking_strategies.build_parser().parse_args(
        [
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
            *extra,
        ]
    )


def _gleif_gb_cell_config(
    roots: WorkspaceRoots,
    args,
    *,
    representation: str = "tfidf",
    similarity_backend: str = "kmeans",
):
    return compare_blocking_strategies._build_cell_config(
        representation=representation,
        similarity_backend=similarity_backend,
        country="gb",
        args=args,
        roots=roots,
        source_descriptor=load_dataset_descriptor(roots=roots, system="gleif"),
        target_descriptor=load_dataset_descriptor(
            roots=roots, system="gb", require_ground_truth=False
        ),
    )


def test_cross_cleanser_naming_the_baseline_profile_is_refused(
    workspace_roots: WorkspaceRoots, capsys
) -> None:
    exit_code = compare_blocking_strategies.main(
        [
            *_roots_argv(workspace_roots),
            "--cleanse-profile",
            "default",
            "--cross-cleanser",
            "default",
            "--dry-run",
        ]
    )

    assert exit_code == 1
    assert "--cross-cleanser 'default' is the --cleanse-profile baseline" in (
        capsys.readouterr().err
    )


def test_dry_run_under_cross_cleanser_reports_both_profiles(
    workspace_roots: WorkspaceRoots, capsys
) -> None:
    exit_code = compare_blocking_strategies.main(
        [
            *_roots_argv(workspace_roots),
            "--cross-cleanser",
            "lowercase",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "cleanse_profiles=['default', 'lowercase']" in out
    assert "would clear" in out
    assert "name_equality_attribution.parquet" in out


def test_load_blocking_run_result_rehydrates_run_keys_and_systems(
    tmp_path: Path,
) -> None:
    """A completed cell's own `summary.json` is the one place its run keys
    and systems can be rehydrated from for a reused, or a
    subprocess-completed, cell."""
    # A real run directory in the tree, since rehydrating one reads back the
    # pairing it was filed under.
    output_dir = resolve_run_location(
        default_workspace_roots(tmp_path),
        source_system="gleif",
        target_system="gb",
        representation="tfidf",
        keys=RunKeys(
            settings="s",
            source_population="p",
            target_population="q",
            truth=None,
            index="i",
        ),
    ).directory
    output_dir.mkdir(parents=True)

    edges_schema = {
        "source_id": pl.Utf8,
        "target_id": pl.Utf8,
        "similarity": pl.Float64,
        "rank": pl.Int64,
        "country": pl.Utf8,
    }
    pl.DataFrame(schema=edges_schema).write_parquet(
        output_dir / "matched_edges.parquet"
    )
    pl.DataFrame(schema=edges_schema).write_parquet(
        output_dir / "raw_matched_edges.parquet"
    )
    pl.DataFrame(
        schema={
            "cluster_id": pl.Utf8,
            "node_id": pl.Utf8,
            "node_name": pl.Utf8,
            "node_role": pl.Utf8,
        }
    ).write_parquet(output_dir / "clusters.parquet")

    summary = {
        "source_system": "gleif",
        "target_system": "gb",
        "keys": {
            "settings_key": "settings",
            "source_population_key": "source-population",
            "target_population_key": "target-population",
            "truth_key": "truth-digest",
            "index_key": "target-index",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = compare_blocking_strategies._load_blocking_run_result(
        default_workspace_roots(tmp_path), output_dir
    )

    recorded = summary["keys"]
    assert isinstance(recorded, dict)
    assert result.keys == RunKeys.from_identity(recorded)
    assert (result.source_system, result.target_system) == ("gleif", "gb")


def test_load_blocking_run_result_leaves_keys_unset_when_the_summary_has_none(
    tmp_path: Path,
) -> None:
    """A `summary.json` with no keys rehydrates with none, never a
    `KeyError`."""
    # A real run directory in the tree, since rehydrating one reads back the
    # pairing it was filed under.
    output_dir = resolve_run_location(
        default_workspace_roots(tmp_path),
        source_system="gleif",
        target_system="gb",
        representation="tfidf",
        keys=RunKeys(
            settings="s",
            source_population="p",
            target_population="q",
            truth=None,
            index="i",
        ),
    ).directory
    output_dir.mkdir(parents=True)

    edges_schema = {
        "source_id": pl.Utf8,
        "target_id": pl.Utf8,
        "similarity": pl.Float64,
        "rank": pl.Int64,
        "country": pl.Utf8,
    }
    pl.DataFrame(schema=edges_schema).write_parquet(
        output_dir / "matched_edges.parquet"
    )
    pl.DataFrame(schema=edges_schema).write_parquet(
        output_dir / "raw_matched_edges.parquet"
    )
    pl.DataFrame(
        schema={
            "cluster_id": pl.Utf8,
            "node_id": pl.Utf8,
            "node_name": pl.Utf8,
            "node_role": pl.Utf8,
        }
    ).write_parquet(output_dir / "clusters.parquet")
    (output_dir / "summary.json").write_text(
        json.dumps({"source_system": "gleif", "target_system": "gb"}), encoding="utf-8"
    )

    result = compare_blocking_strategies._load_blocking_run_result(
        default_workspace_roots(tmp_path), output_dir
    )

    assert result.keys is None


def test_a_cell_is_keyed_by_the_identity_its_run_blocking_subprocess_builds(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, monkeypatch
) -> None:
    """A cell's comparison row describes the configuration the script built,
    which holds only while that configuration is the one `run_blocking.py`
    builds from the argv and environment the cell hands it, the one its run is
    keyed by."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    for name, value in compare_blocking_strategies._cell_environment(
        workspace_roots
    ).items():
        if name in ENVIRONMENT_VARIABLES:
            monkeypatch.setenv(name, value)
    # Every profile off its default, so each is shown to reach the subprocess.
    args = compare_blocking_strategies._with_preprocess_profile(
        compare_blocking_strategies._with_cleanse_profile(
            _gleif_gb_args("--tokenizer-profile", "naive"), "lowercase"
        ),
        "default|-company_type",
    )

    backend_for = {"tfidf": "kmeans", "wordpiece": "kmeans", "sbert": "hnsw"}
    for representation in ("tfidf", "wordpiece", "sbert"):
        similarity_backend = backend_for[representation]
        config = _gleif_gb_cell_config(
            workspace_roots,
            args,
            representation=representation,
            similarity_backend=similarity_backend,
        )
        argv = compare_blocking_strategies._build_cell_argv(
            representation=representation,
            similarity_backend=similarity_backend,
            source_system="gleif",
            target_system="gb",
            country="gb",
            args=args,
        )
        run_args = run_blocking.build_parser().parse_args(argv[2:])
        run_config = run_blocking._build_config(
            run_args, strategy=run_blocking._resolve_strategy(run_args)
        )

        assert blocking_run_settings(config) == blocking_run_settings(run_config)


@pytest.mark.integration
def test_a_cross_cleanser_sweep_attributes_the_moved_pair_and_is_reused_when_rerun(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    argv = [
        *_roots_argv(workspace_roots),
        "--source",
        "gleif",
        "--target",
        "gb",
        "--countries",
        "gb",
        "--representations",
        "tfidf",
        "--cross-cleanser",
        "default|-lowercase",
    ]
    assert compare_blocking_strategies.main(argv) == 0
    capsys.readouterr()

    assert compare_blocking_strategies.main(argv) == 0

    # Two cleanse profiles, each measured on both mandatory backends for the
    # representation, `kmeans` and `lsh`.
    assert capsys.readouterr().out.count("reused the finished run") == 4
    pairing_dir = resolve_pairing_dir(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    assert len(list(pairing_dir.glob("tfidf/*/manifest.json"))) == 4
    comparison_location = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    comparison = pl.read_parquet(comparison_location.path(ComparisonArtefact.REPORT))
    assert set(comparison.get_column("name_transform")) == {
        "cleanse:default;preprocess:default",
        "cleanse:default|-lowercase;preprocess:default",
    }
    # Without lowercasing, the company-type step no longer reads `Acme Limited`'s
    # legal form, so the pair equal once cleansed under the default profile is
    # no longer equal as cleansed. Preprocessing still removes the company
    # type before scoring, so both sides reach the scan as one string: the
    # pair lands at `preprocessed`, not `never`, and nothing is credited to
    # the algorithm for finding it.
    attribution = pl.read_parquet(
        comparison_location.path(ComparisonArtefact.NAME_EQUALITY_ATTRIBUTION)
    )
    # One row per backend, since each finds its own pairs.
    assert attribution.select(
        "similarity_backend",
        "source_id",
        "baseline_name_equality",
        "variant_name_equality",
        "never_attribution",
    ).sort("similarity_backend").rows() == [
        ("kmeans", "gleif:1", "cleansed", "preprocessed", None),
        ("lsh", "gleif:1", "cleansed", "preprocessed", None),
    ]
