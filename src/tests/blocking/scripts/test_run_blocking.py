from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from company_vectorize.lsh_similarity import (
    DEFAULT_LSH_NUM_BANDS,
    DEFAULT_LSH_NUM_PERM,
    DEFAULT_LSH_SEED,
)
from company_vectorize.partition_similarity import (
    DEFAULT_KMEANS_BATCH_RESEED_BELOW,
    DEFAULT_KMEANS_CLUSTERS,
    DEFAULT_KMEANS_FIT_ROWS,
    DEFAULT_KMEANS_MAX_PASSES,
    DEFAULT_KMEANS_RESTARTS,
    DEFAULT_KMEANS_SEED,
    DEFAULT_KMEANS_STALL_BATCHES,
    DEFAULT_KMEANS_START,
    DEFAULT_KMEANS_START_ROWS,
    DEFAULT_KMEANS_STOP_TOLERANCE,
)

from blocking.contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingStrategyConfig,
    blocking_settings_key,
)
from blocking.run_layout import iter_blocking_runs, resolve_pairing_dir
from blocking.truth import MatchedLayerTruth
from scripts import run_blocking
from scripts.cli_common import applicable_settings_record
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    layer_directory,
    perturbed_dataset_dir,
)
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots, default_workspace_roots


def _roots_argv(roots: WorkspaceRoots) -> list[str]:
    return [
        "--data-dir",
        str(roots.data),
        "--output-dir",
        str(roots.artifacts),
        "--temp-dir",
        str(roots.temp),
    ]


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
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_dir_for,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )


@pytest.mark.integration
def test_run_blocking_script_writes_artefacts(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    """The real run, kept because what it asserts is this script's own: the
    manifest recording the applicable settings, the evaluation lines it
    prints, and a repeated run finding the finished one by its keys and
    recording it for the caller, none of which a dry run reaches."""
    _write_gleif_gb_fixture(layer_fixture_dir)

    argv = [
        *_roots_argv(workspace_roots),
        "--source",
        "gleif",
        "--target",
        "gb",
        "--countries",
        "gb",
        "--top-k",
        "2",
        "--min-similarity",
        "0.2",
        "--additional-args",
        "tfidf.ngram_min=1",
        "tfidf.ngram_max=2",
    ]
    exit_code = run_blocking.main(argv)

    assert exit_code == 0
    assert "[blocking] eval-summary country=gb" in capsys.readouterr().out
    (run,) = list(iter_blocking_runs(workspace_roots))
    run_dir = run.directory
    # Row-level data as parquet; everything summary-shaped in summary.json.
    assert (run_dir / "matched_edges.parquet").exists()
    assert (run_dir / "raw_matched_edges.parquet").exists()
    assert (run_dir / "clusters.parquet").exists()
    assert not (run_dir / "pruning_summary.parquet").exists()
    assert not (run_dir / "pair_truth_eval.parquet").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {
        key: value for key, value in manifest["identity"].items() if key != "settings"
    } == summary["keys"]
    assert "commit" in manifest
    configuration = manifest["configuration"]
    assert configuration["top_k"] == {
        "value": 2,
        "source": "given",
        "surface": "--top-k",
    }
    assert configuration["tfidf_ngram_min"]["value"] == 1
    assert configuration["similarity_backend"]["source"] == "default"
    # A tfidf run reads no tokenizer, so no tokenizer setting applied to it.
    assert "tokenizer" not in configuration

    # The same run again finds the finished one by its keys before scoring,
    # reuses it, and records where it is for the caller.
    record = workspace_roots.temp / "record.json"
    record.parent.mkdir(parents=True)
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    assert run_blocking.main([*argv, "--run-record", str(record)]) == 0
    assert json.loads(record.read_text(encoding="utf-8")) == {
        "run_dir": str(run_dir),
        "reused": True,
    }
    assert (run_dir / "manifest.json").read_bytes() == manifest_bytes


@pytest.mark.integration
def test_run_blocking_ground_truth_false_completes_with_no_truth_scoring(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    """`--ground-truth false` loads the source from `cleansed` with no
    ground truth, and the run's own summary says evaluation was skipped
    rather than leaving that only implied by an absent countries block."""
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gleif:1", "name": "acme limited", "jurisdiction_code": "gb"}
        ],
    )
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )

    exit_code = run_blocking.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
            "--ground-truth",
            "false",
            "--additional-args",
            "tfidf.ngram_min=1",
            "tfidf.ngram_max=2",
        ]
    )

    assert exit_code == 0
    assert "[blocking] eval-summary" not in capsys.readouterr().out
    (run,) = list(iter_blocking_runs(workspace_roots))
    summary = json.loads((run.directory / "summary.json").read_text(encoding="utf-8"))

    assert summary["evaluation_skipped"] is True
    assert summary["countries"] is None


def test_run_blocking_help_names_each_default(capsys) -> None:
    with pytest.raises(SystemExit):
        run_blocking.build_parser().parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())

    assert "(default: 20)" in help_text
    assert "(default: 1000)" in help_text
    assert "tfidf.ngram_min (default: 2)" in help_text
    assert "--run-record" not in help_text


def test_run_blocking_dry_run_reports_resolved_config_without_executing(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    """--dry-run resolves and validates the run config, reading its inputs
    under --data-dir and naming where the run would land under --output-dir,
    and stops before `execute_blocking_run` -- no artefact is written."""
    _write_gleif_gb_fixture(layer_fixture_dir)

    exit_code = run_blocking.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
            "--source-chunk-size",
            "1",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run] run_blocking: would proceed with:" in out
    assert "match_col='match_uri'" in out
    assert "name_transform='cleanse'" in out
    assert "source_system='gleif' (given)" in out
    assert "top_k=20 (default)" in out
    assert "source_chunk_size=1 (given)" in out
    assert "tokenizer" not in out
    assert "kmeans_clusters" not in out
    assert f"source_dir={layer_fixture_dir('gleif', layer='matched')!r}" in out
    target_dir = layer_fixture_dir("gb", layer="canonical") / "2026-01-01"
    assert f"target_dir={target_dir!r}" in out
    pairing_dir = resolve_pairing_dir(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    assert f"would extend {pairing_dir}" in out
    assert not workspace_roots.artifacts.exists()


def test_run_blocking_dry_run_files_a_perturbed_selector_under_its_system(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    """`--source ie --perturbed p1` is the whole of what a perturbed run needs:
    the version and seed are the ones on disk, the truth column is implied, and
    the run is filed under the target and the `perturbed` kind."""
    perturbed_layer = layer_directory(
        perturbed_dataset_dir(
            roots=workspace_roots,
            source_system="ie",
            profile_id="p1",
            version="v1",
            seed=1,
        ),
        CLEANSED_LAYER_NAME,
    )
    partition_dir = layer_partition_dir(perturbed_layer, value="ie")
    partition_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "system_uri": ["perturbed://ie/p1/1/a"],
            "source_uri": ["ie://1"],
            "match_uri": [None],
            "name": ["acme limted"],
            "jurisdiction_code": ["ie"],
        }
    ).write_parquet(partition_dir / "part-00000.parquet")
    _write_partition(
        layer_fixture_dir,
        system="ie",
        layer="canonical",
        country="ie",
        rows=[
            {"system_uri": "ie://1", "name": "acme limited", "jurisdiction_code": "ie"}
        ],
    )

    exit_code = run_blocking.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "ie",
            "--perturbed",
            "p1",
            "--target",
            "ie",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "has_ground_truth=True" in out
    pairing_dir = resolve_pairing_dir(
        workspace_roots, source_system="perturbed://ie/p1/v1/1", target_system="ie"
    )
    assert pairing_dir.parts[-3:] == ("ie", "perturbed", "ie_p1_v1_1")
    assert f"would extend {pairing_dir}" in out


def _gate_args(*extra: str):
    return run_blocking.build_parser().parse_args(
        ["--source", "gleif", "--target", "gb", *extra]
    )


def test_run_blocking_additional_args_reach_strategy_config() -> None:
    strategy = run_blocking._resolve_strategy(
        _gate_args("--additional-args", "pruning.max_candidates_per_target=1")
    )

    assert strategy.max_candidates_per_target == 1


def test_run_blocking_sbert_model_flag_defaults_to_none() -> None:
    strategy = run_blocking._resolve_strategy(_gate_args("--representation", "sbert"))

    assert strategy.sbert_model_name is None


def test_run_blocking_sbert_model_flag_reaches_strategy_config() -> None:
    strategy = run_blocking._resolve_strategy(
        _gate_args(
            "--representation", "sbert", "--sbert-model", "fr-sentence-camembert-base"
        )
    )

    assert strategy.sbert_model_name == "fr-sentence-camembert-base"


def test_run_blocking_target_neighbor_flags_default_to_none() -> None:
    strategy = run_blocking._resolve_strategy(_gate_args())

    assert strategy.target_neighbor_min_similarity is None
    assert strategy.target_neighbor_max_per_target is None


def test_run_blocking_target_neighbor_flags_reach_strategy_config() -> None:
    strategy = run_blocking._resolve_strategy(
        _gate_args(
            "--target-neighbor-min-similarity",
            "0.85",
            "--target-neighbor-max-per-target",
            "10",
        )
    )

    assert strategy.target_neighbor_min_similarity == 0.85
    assert strategy.target_neighbor_max_per_target == 10


def test_run_blocking_backend_options_coerce_bool_and_numeric_values() -> None:
    strategy = run_blocking._resolve_strategy(
        _gate_args(
            "--additional-args",
            "backend.prefix_filter=false",
            "backend.top_k=5",
            "backend.ratio=0.5",
        )
    )

    assert strategy.backend_options == {
        "prefix_filter": False,
        "top_k": 5,
        "ratio": 0.5,
    }
    assert strategy.backend_options["prefix_filter"] is False


def test_run_blocking_similarity_backend_accepts_partition_backends() -> None:
    # --similarity-backend's choices must include kmeans/hdbscan
    # (previously hardcoded to sklearn/sparse_dot_topn/svd_rerank), and
    # backend.<key> options (kmeans_clusters/min_cluster_size) must reach
    # BlockingStrategyConfig.backend_options through the same generic
    # --additional-args accumulation prefix_filter already uses.
    for backend, option_token in (
        ("kmeans", "backend.kmeans_clusters=10"),
        ("hdbscan", "backend.min_cluster_size=3"),
    ):
        strategy = run_blocking._resolve_strategy(
            _gate_args(
                "--similarity-backend", backend, "--additional-args", option_token
            )
        )
        assert strategy.similarity_backend == backend
        key, _, value = option_token.removeprefix("backend.").partition("=")
        assert strategy.backend_options is not None
        assert strategy.backend_options[key] == int(value)


def test_run_blocking_backend_run_carries_its_settings_given_or_defaulted() -> None:
    """A run names every setting of its backend in its options whether given
    or defaulted, so two runs that differ only by a default never share a
    key; a backend with no declared settings carries none."""
    defaulted = run_blocking._resolve_strategy(
        _gate_args("--similarity-backend", "kmeans")
    )
    assert defaulted.backend_options == {
        "kmeans_clusters": DEFAULT_KMEANS_CLUSTERS,
        "kmeans_start": DEFAULT_KMEANS_START,
        "kmeans_start_rows": DEFAULT_KMEANS_START_ROWS,
        "kmeans_fit_rows": DEFAULT_KMEANS_FIT_ROWS,
        "kmeans_restarts": DEFAULT_KMEANS_RESTARTS,
        "kmeans_batch_reseed_below": DEFAULT_KMEANS_BATCH_RESEED_BELOW,
        "max_passes": DEFAULT_KMEANS_MAX_PASSES,
        "stop_tolerance": DEFAULT_KMEANS_STOP_TOLERANCE,
        "stall_batches": DEFAULT_KMEANS_STALL_BATCHES,
        "seed": DEFAULT_KMEANS_SEED,
    }

    other_default = replace(
        defaulted,
        backend_options={**(defaulted.backend_options or {}), "kmeans_clusters": 50},
    )
    assert _settings_key(defaulted) != _settings_key(other_default)

    lsh = run_blocking._resolve_strategy(_gate_args("--similarity-backend", "lsh"))
    assert lsh.backend_options == {
        "num_perm": DEFAULT_LSH_NUM_PERM,
        "num_bands": DEFAULT_LSH_NUM_BANDS,
        "seed": DEFAULT_LSH_SEED,
    }

    unsettled = run_blocking._resolve_strategy(
        _gate_args("--similarity-backend", "sparse_dot_topn")
    )
    assert unsettled.backend_options is None


def _settings_key(strategy: BlockingStrategyConfig) -> str:
    def descriptor(system: str) -> BlockingDatasetDescriptor:
        return BlockingDatasetDescriptor(
            system=system,
            system_dir=Path(f"/data/{system}/cleansed"),
            layer="cleansed",
            has_ground_truth=False,
            matched_target_systems=(),
            available_countries=("gb",),
        )

    return blocking_settings_key(
        BlockingRunConfig(
            roots=default_workspace_roots(Path("/root")),
            prepared_base_dir=None,
            source=descriptor("gleif"),
            target=descriptor("gb"),
            countries=None,
            strategy=strategy,
            truth=MatchedLayerTruth(),
        )
    )


def test_run_blocking_backend_seed_reaches_each_backend_and_keys_the_run() -> None:
    """`backend.seed` is the one `backend_seed` setting on a kmeans run and
    on an lsh run, and reaches that backend's options as `seed`. A named seed
    changes the run's key."""
    for backend in ("kmeans", "lsh"):
        seeded_args = _gate_args(
            "--similarity-backend", backend, "--additional-args", "backend.seed=7"
        )
        resolved, _ = run_blocking._resolve_settings(seeded_args)
        record = applicable_settings_record(resolved)
        assert record["backend_seed"]["value"] == 7
        assert record["backend_seed"]["surface"] == "backend.seed"

        seeded = run_blocking._resolve_strategy(seeded_args)
        unseeded = run_blocking._resolve_strategy(
            _gate_args("--similarity-backend", backend)
        )
        assert seeded.backend_options is not None
        assert seeded.backend_options["seed"] == 7
        assert (unseeded.backend_options or {}).get("seed") == DEFAULT_KMEANS_SEED
        assert _settings_key(seeded) != _settings_key(unseeded)


def test_run_blocking_similarity_backend_accepts_dense_brute() -> None:
    strategy = run_blocking._resolve_strategy(
        _gate_args(
            "--similarity-backend",
            "dense_brute",
            "--additional-args",
            "backend.storage_dtype=float32",
            "backend.block_bytes=1024",
        )
    )

    assert strategy.similarity_backend == "dense_brute"
    assert strategy.backend_options == {
        "storage_dtype": "float32",
        "block_bytes": 1024,
    }


def test_run_blocking_main_returns_error_exit_code_for_missing_system(
    workspace_roots: WorkspaceRoots, capsys
) -> None:
    exit_code = run_blocking.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[blocking] error:" in captured.err


def test_run_blocking_refuses_no_ground_truth_for_the_default_match_column(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    """`--ground-truth true` (the default) requires a `matched` layer under
    `match_uri`; a source with only a `cleansed` layer has none, and is
    refused before any run directory is written, naming the system, the
    layer it needed and had none of, and the system's own data directory it
    looked under."""
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme", "jurisdiction_code": "gb"}],
    )
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )

    exit_code = run_blocking.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
        ]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "gleif" in stderr
    assert "matched" in stderr
    assert str(workspace_roots.data / "gleif") in stderr
    assert not workspace_roots.artifacts.exists()


def test_run_blocking_refuses_no_ground_truth_for_another_match_column(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    """`--match-col source_uri` reads a `ColumnTruth`; a resolved layer
    missing that column is refused the same way, naming the column and the
    layer directory it was missing from, before any run directory is
    written."""
    cleansed_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme", "jurisdiction_code": "gb"}],
    )
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )

    exit_code = run_blocking.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--match-col",
            "source_uri",
        ]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "'source_uri'" in stderr
    assert str(cleansed_dir) in stderr
    assert not workspace_roots.artifacts.exists()


def test_run_blocking_gate_flags_default_to_no_backend_options() -> None:
    # The dense-vocabulary gate's options are written only when they move off
    # the default, so adding these flags does not change the run identity; the
    # prefix filter's resolved value is always written where it applies, so a
    # filtered run never shares an unfiltered run's identity.
    strategy = run_blocking._resolve_strategy(_gate_args())

    assert strategy.backend_options == {"prefix_filter": False}


def test_run_blocking_prefix_filter_is_absent_where_it_does_not_apply() -> None:
    for extra in (
        ["--representation", "sbert"],
        ["--similarity-backend", "sparse_dot_topn"],
    ):
        strategy = run_blocking._resolve_strategy(_gate_args(*extra))

        assert strategy.backend_options is None, extra


def test_run_blocking_force_and_max_rows_reach_backend_options() -> None:
    strategy = run_blocking._resolve_strategy(
        _gate_args("--force", "true", "--max-rows", "500")
    )

    assert strategy.backend_options == {
        "prefix_filter": False,
        "force": True,
        "max_rows": 500,
    }


def test_run_blocking_gate_flags_coexist_with_backend_additional_args() -> None:
    strategy = run_blocking._resolve_strategy(
        _gate_args("--force", "true", "--additional-args", "backend.prefix_filter=true")
    )

    assert strategy.backend_options == {"prefix_filter": True, "force": True}


def test_run_blocking_refuses_gated_combination_with_a_readable_error(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
) -> None:
    # End-to-end through the script's own error handling: the fixture has no
    # trained wordpiece tokenizer at all, so reaching a clean exit code 1 with
    # the refusal message is itself evidence the gate fired before any
    # tokenization or index building was attempted.
    _write_gleif_gb_fixture(layer_fixture_dir)

    exit_code = run_blocking.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--countries",
            "gb",
            "--representation",
            "wordpiece",
            "--max-rows",
            "0",
        ]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "wordpiece" in stderr
    assert "sklearn" in stderr
    assert "--force" in stderr
    assert "--max-rows" in stderr
    assert not workspace_roots.artifacts.exists()
