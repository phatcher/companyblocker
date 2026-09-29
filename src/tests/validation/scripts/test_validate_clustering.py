from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from acquisition.cleanser_orchestrate import materialize_cleansed_merge
from acquisition.constants_status import STATUS_RESEARCH_REQUIRED
from scripts import validate_clustering
from workspace.artifact_layout import validation_artifact_root
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_cleansed_chunk(
    roots: WorkspaceRoots, system: str, rows: list[dict[str, object]]
) -> None:
    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    source_dir = cleansed_dir / "chunks"
    source_dir.mkdir(parents=True, exist_ok=True)
    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        cleansed_value = str(enriched.get("name_cleansed") or "")
        enriched.setdefault("name", cleansed_value)
        enriched.setdefault("name_cleansed_basic", cleansed_value)
        enriched_rows.append(enriched)
    pl.DataFrame(enriched_rows).write_parquet(source_dir / f"{system}-001.parquet")
    materialize_cleansed_merge(
        source_dir=source_dir,
        output_dir=cleansed_dir,
        system=system,
        name_col="name_cleansed",
        rows_per_file=1_000_000,
        force_rebuild=True,
    )


@pytest.mark.integration
def test_validate_clustering_script_writes_artifacts(
    tmp_path: Path, workspace_roots: WorkspaceRoots
) -> None:
    """The full pipeline actually writes the five artifact files a run
    produces; a dry run stops before any of them exist, so this needs the
    real call."""
    run_date = "2026-07-07"
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
        ],
    )

    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "--targets",
            "gb",
            "--countries",
            "gb",
            "--name-col",
            "name_cleansed",
            "--date",
            run_date,
            "--output-dir",
            "artifacts/validation/test-run",
            "--top-k",
            "1",
            "--min-similarity",
            "0.2",
        ]
    )
    exit_code = validate_clustering.run_validation(args)

    assert exit_code == 0
    run_dir = validation_artifact_root(workspace_roots) / "test-run"
    assert (run_dir / "source_outcomes.parquet").exists()
    assert (run_dir / "clusters.parquet").exists()
    assert (run_dir / "directional_coverage.parquet").exists()
    assert (run_dir / "cluster_shape.parquet").exists()
    assert (run_dir / "pair_truth_eval.parquet").exists()


@pytest.mark.integration
def test_validate_clustering_prepare_only_materializes_all_source_jurisdictions(
    tmp_path: Path, workspace_roots: WorkspaceRoots
) -> None:
    """`prepare_system_dataset` actually writing a partition per jurisdiction
    the source system carries, not only the ones the target shares, is the
    real work's own output; a dry run resolves no partitions to check."""
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gleif:2",
                "name_cleansed": "beta holdings",
                "jurisdiction_code": "fr",
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
        ],
    )

    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "--targets",
            "gb",
            "--name-col",
            "name_cleansed",
            "--prepare-only",
        ]
    )
    exit_code = validate_clustering.run_validation(args)

    assert exit_code == 0
    assert (
        system_layer_dir(workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME)
        / "primary"
        / "jurisdiction_code=gb"
    ).exists()
    assert (
        system_layer_dir(workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME)
        / "primary"
        / "jurisdiction_code=fr"
    ).exists()
    assert (
        system_layer_dir(workspace_roots, "gb", layer=CLEANSED_LAYER_NAME)
        / "primary"
        / "jurisdiction_code=gb"
    ).exists()


def test_validate_clustering_prepare_only_filters_non_live_systems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_filter_live_systems` (see its own direct tests below) already covers
    the skip decision itself; what only the real `--prepare-only` flow was
    covering was that the wiring reaches it and reports the skip before
    anything is prepared -- a dry run reaches the same point and needs no
    cleansed fixture to prove it."""
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "dbpedia",
            "--targets",
            "gb",
            "dbpedia",
            "--name-col",
            "name_cleansed",
            "--prepare-only",
            "--dry-run",
        ]
    )
    exit_code = validate_clustering.run_validation(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "skipping non-live source system 'dbpedia'" in out
    assert "skipping non-live target system 'dbpedia'" in out
    assert "source_systems=('gleif',)" in out
    assert "target_systems=('gb',)" in out


def test_validate_clustering_prepare_only_allows_research_when_enabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin the status rather than relying on whichever catalog entry happens to
    # be research_required. This test previously drove `dbpedia` because it
    # carried that status; `dbpedia` is now `stopped`, which no --allow-research
    # opt-in can unlock (resolve_runnable_statuses never admits it), and no
    # remaining system is both research_required and cleanse-allowed -- `edgar`
    # is research_required but its policy lists `acquire` only. The branch under
    # test is "research_required plus an allowed stage is admitted", so the
    # status is fixed here and the rest of the plan stays real.
    real_get_system_plan = validate_clustering.get_system_plan

    def research_required_dbpedia(system_code: str):
        plan = real_get_system_plan(system_code)
        if plan.code != "dbpedia":
            return plan
        # Both gates have to be opened: the status admits the system into a
        # research run at all, and allow_research_runtime plus an allowed
        # stage is the per-stage policy on top of it. dbpedia's catalog entry
        # already lists `cleanse` among allowed_stages but has the runtime
        # flag off, so flipping status alone still denies the stage.
        assert plan.research is not None
        return replace(
            plan,
            status=STATUS_RESEARCH_REQUIRED,
            research=replace(plan.research, allow_research_runtime=True),
        )

    monkeypatch.setattr(
        validate_clustering, "get_system_plan", research_required_dbpedia
    )

    # No cleansed fixture is needed: a dry run resolves which systems are live
    # and reports the plan without preparing or scoring anything.
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "dbpedia",
            "--targets",
            "gb",
            "--name-col",
            "name_cleansed",
            "--prepare-only",
            "--allow-research",
            "--dry-run",
        ]
    )
    exit_code = validate_clustering.run_validation(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "skipping" not in out
    assert "source_systems=('dbpedia',)" in out
    assert "target_systems=('gb',)" in out


def test_validate_clustering_prepare_only_accepts_all_token(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The special 'all' token is resolved to concrete systems and reported
    before anything is prepared -- a dry run reaches that point without
    needing `prepare_system_dataset` mocked out from under it."""
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "all",
            "--targets",
            "ie",
            "--name-col",
            "name_cleansed",
            "--prepare-only",
            "--dry-run",
        ]
    )

    exit_code = validate_clustering.run_validation(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[info] Resolved systems for --sources all:" in out


def test_validate_clustering_rejects_non_positive_max_rows(tmp_path: Path) -> None:
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "--targets",
            "gb",
            "--name-col",
            "name_cleansed",
            "--prepare-only",
            "--max-rows",
            "0",
        ]
    )

    try:
        validate_clustering.run_validation(args)
        assert False, "Expected ValueError for non-positive --max-rows"
    except ValueError as exc:
        assert str(exc) == "max_rows must be greater than 0"


def test_validate_clustering_parser_rejects_hybrid_text_view() -> None:
    parser = validate_clustering.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--sources",
                "gleif",
                "--targets",
                "gb",
                "--text-view",
                "hybrid",
            ]
        )


@pytest.mark.integration
def test_validate_clustering_uses_method_hashed_default_output(
    tmp_path: Path, workspace_roots: WorkspaceRoots
) -> None:
    """The default output directory names a real, completed run's own
    written files (`run_manifest.json`, `source_outcomes.parquet`); a dry
    run reports where they would go without writing any of them."""
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "ie",
            }
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "ie",
        [
            {
                "system_uri": "ie:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "ie",
            }
        ],
    )

    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "--targets",
            "ie",
            "--name-col",
            "name_cleansed",
            "--max-rows",
            "10",
            "--additional-args",
            "tfidf.ngram_min=1",
            "tfidf.ngram_max=2",
        ]
    )

    exit_code = validate_clustering.run_validation(args)
    assert exit_code == 0

    pair_root = (
        validation_artifact_root(workspace_roots)
        / "clustering"
        / "gleif__ie"
        / "tfidf"
        / "cosine-knn_cc"
        / "ngram-1-2_analyzer-char_wb_backend-sklearn"
    )
    assert pair_root.exists()
    method_runs = [child for child in pair_root.iterdir() if child.is_dir()]
    assert len(method_runs) == 1
    run_dir = method_runs[0]
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "source_outcomes.parquet").exists()


def test_validate_clustering_uses_representation_family_output_for_tokens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`method_id`'s family-config segment carries the tokenizer settings a
    caller gave -- resolved before any scoring happens, so a dry run reports
    it without a fake `run_validation_matrix` or a real write to check."""
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "--targets",
            "ie",
            "--name-col",
            "name_cleansed",
            "--representation",
            "wordpiece",
            "--additional-args",
            "tokenizer.name=wordpiece",
            "tokenizer.scope=global",
            "noise_words.profile=aggressive",
            "noise_words.set_kind=combined",
            "--dry-run",
        ]
    )

    exit_code = validate_clustering.run_validation(args)
    assert exit_code == 0

    out = capsys.readouterr().out
    assert (
        "method_id='wordpiece_cosine-knn_cc_scope-global_stop-aggressive-combined_backend-sklearn'"
        in out
    )


def _wordpiece_runtime_context(
    roots: WorkspaceRoots, *, extra_tokenizer_args: list[str]
):
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(roots.checkout),
            "--sources",
            "gleif",
            "--targets",
            "ie",
            "--representation",
            "wordpiece",
            "--additional-args",
            "tokenizer.name=wordpiece",
            *extra_tokenizer_args,
        ]
    )
    return validate_clustering._build_validation_runtime_context(
        args=args,
        roots=roots,
        source_systems=("gleif",),
        target_systems=("ie",),
    )


def test_validate_clustering_defaults_to_the_tokenizer_the_profile_names(
    workspace_roots: WorkspaceRoots,
) -> None:
    config, _, _, _, family_config_slug, effective_method_config, _ = (
        _wordpiece_runtime_context(workspace_roots, extra_tokenizer_args=[])
    )

    assert config.tokenizer_path is None
    assert "path" not in effective_method_config["tokenizer"]
    assert (
        family_config_slug == "scope-country_stop-aggressive-combined_backend-sklearn"
    )


def test_validate_clustering_threads_a_tokenizer_given_by_path_through(
    workspace_roots: WorkspaceRoots,
) -> None:
    config, _, _, _, _, effective_method_config, _ = _wordpiece_runtime_context(
        workspace_roots, extra_tokenizer_args=["tokenizer.path=candidate/model.json"]
    )

    assert config.tokenizer_path == "candidate/model.json"
    assert effective_method_config["tokenizer"]["path"] == "candidate/model.json"


def test_validate_clustering_a_run_given_a_tokenizer_path_lands_beside_the_default_run(
    workspace_roots: WorkspaceRoots,
) -> None:
    default_config, _, _, _, _, _, _ = _wordpiece_runtime_context(
        workspace_roots, extra_tokenizer_args=[]
    )
    override_config, _, _, _, _, _, _ = _wordpiece_runtime_context(
        workspace_roots, extra_tokenizer_args=["tokenizer.path=candidate/model.json"]
    )

    assert default_config.output_dir != override_config.output_dir


def test_validate_clustering_rejects_tokenizer_args_with_name_mode(
    tmp_path: Path,
) -> None:
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "--targets",
            "ie",
            "--additional-args",
            "tokenizer.name=wordpiece",
        ]
    )

    with pytest.raises(ValueError, match="tokenizer"):
        validate_clustering.run_validation(args)


def test_validate_clustering_rejects_tfidf_args_with_tokens_mode(
    tmp_path: Path,
) -> None:
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "gleif",
            "--targets",
            "ie",
            "--representation",
            "wordpiece",
            "--additional-args",
            "tfidf.ngram_min=1",
        ]
    )

    with pytest.raises(ValueError, match="tfidf"):
        validate_clustering.run_validation(args)


def test_perturbed_reads_every_source_as_its_dataset_pinned_to_what_is_on_disk(
    workspace_roots: WorkspaceRoots,
) -> None:
    from workspace.data_layout import perturbed_dataset_dir

    perturbed_dataset_dir(
        workspace_roots,
        source_system="gb",
        profile_id="robustness",
        version="v1",
        seed=7,
    ).mkdir(parents=True)
    args = validate_clustering.build_parser().parse_args(
        ["--sources", "gb", "--targets", "gb", "--perturbed", "robustness",
         "--root", str(workspace_roots.checkout)]
    )  # fmt: skip

    _, sources, targets = validate_clustering._resolve_selected_live_systems(args)

    assert sources == ("perturbed://gb/robustness/v1/7",)
    assert targets == ("gb",)


def test_filter_live_systems_rejects_unregistered_non_perturbed_system() -> None:
    with pytest.raises(ValueError, match="Unknown system 'not-a-real-system'"):
        validate_clustering._filter_live_systems(
            ("not-a-real-system",), allow_research=False
        )


def test_resolve_output_slugifies_colon_bearing_perturbed_selector(
    tmp_path: Path, workspace_roots: WorkspaceRoots
) -> None:
    # ':' is illegal in a Windows path segment (reserved for drive letters) -- a
    # "perturbed:<profile_id>" selector must not reach Path.mkdir() unslugified.
    args = validate_clustering.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--sources",
            "perturbed:robustness-v1",
            "--targets",
            "gb",
        ]
    )

    output_dir, _ = validate_clustering._resolve_output(
        args=args,
        roots=workspace_roots,
        source_systems=("perturbed:robustness-v1",),
        target_systems=("gb",),
        representation_family="tfidf",
        comparison="cosine-knn_cc",
        family_config_slug="ngram-2-3_backend-sklearn",
        method_config={"clustering": "knn_cc"},
    )

    # Check the constructed pair-label segment specifically, not the full path -- an absolute
    # Windows path legitimately starts with a drive letter's own ':'.
    pair_label_segment = output_dir.relative_to(
        validation_artifact_root(workspace_roots) / "clustering"
    ).parts[0]
    assert ":" not in pair_label_segment
    assert pair_label_segment == "perturbed-robustness-v1__gb"
