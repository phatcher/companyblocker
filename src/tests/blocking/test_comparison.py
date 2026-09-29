import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from blocking.comparison import (
    StrategyRunEntry,
    _encode_accelerator_settings,
    _is_exact_backend,
    build_strategy_comparison,
    check_strategy_comparison_repeatability,
    diff_pair_recovery,
)
from blocking.contracts import BlockingRunConfig, BlockingStrategyConfig
from blocking.loader import load_dataset_descriptor
from blocking.reporting import write_strategy_comparison_report
from blocking.workflow import execute_blocking_run
from tests.promoted_tokenizers import promoted_tokenizer_files
from workspace.layer_layout import layer_partition_dir
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
    roots: WorkspaceRoots,
    *,
    representation: str = "tfidf",
    tokenizer: str = "wordpiece",
    tokenizer_profile: str = "promoted",
    similarity_backend: str = "sklearn",
    require_source_ground_truth: bool = True,
    backend_options: dict[str, object] | None = None,
    exact_name_filter: bool = True,
    name_transform: str = "identity",
) -> BlockingRunConfig:
    source = load_dataset_descriptor(
        roots=roots, system="gleif", require_ground_truth=require_source_ground_truth
    )
    target = load_dataset_descriptor(
        roots=roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation=representation,
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        similarity_backend=similarity_backend,
        backend_options=backend_options,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
        tokenizer=tokenizer,
        tokenizer_profile=tokenizer_profile,
        exact_name_filter=exact_name_filter,
        name_transform=name_transform,
    )
    return BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
    )


def _fake_tokenize(frame, cfg, *, name_col="name"):
    return frame.with_columns(
        pl.col(name_col)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.split(" ")
        .alias("blocking_tokens")
    )


def _write_placeholder_tokenizers(roots: WorkspaceRoots, *systems: str) -> None:
    """A token-scoring run records the content hash of the tokenizer file each
    side read, whether or not the tokenize step itself is faked, so the file
    has to exist at the path the run resolves."""
    for system in systems:
        path = promoted_tokenizer_files(roots, system=system).model
        path.write_text(f"{system} placeholder", encoding="utf-8")


def test_build_strategy_comparison_places_two_strategies_side_by_side(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, mocker
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    _write_placeholder_tokenizers(workspace_roots, "gleif", "gb")
    tfidf_config = _build_config(workspace_roots, representation="tfidf")
    wordpiece_config = _build_config(workspace_roots, representation="wordpiece")

    mocker.patch("blocking.workflow._tokenize", side_effect=_fake_tokenize)

    tfidf_result = execute_blocking_run(tfidf_config)
    wordpiece_result = execute_blocking_run(wordpiece_config)

    comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="tfidf",
                config=tfidf_config,
                result=tfidf_result,
                runtime_seconds=1.5,
            ),
            StrategyRunEntry(
                label="wordpiece",
                config=wordpiece_config,
                result=wordpiece_result,
                runtime_seconds=2.5,
            ),
        ]
    )

    assert set(comparison.get_column("label").unique().to_list()) == {
        "tfidf",
        "wordpiece",
    }
    assert set(comparison.get_column("stage").unique().to_list()) == {"pruned", "raw"}

    # The universe is populated for every run; a level can rightly be empty,
    # and an empty population has no precision to report.
    pruned = comparison.filter(
        (pl.col("stage") == "pruned") & (pl.col("population") == "universe")
    )
    for column in (
        "precision",
        "recall",
        "reduction_ratio",
        "recall_at_k",
        "candidate_set_size_ratio",
    ):
        assert pruned.get_column(column).null_count() == 0

    tfidf_rows = pruned.filter(pl.col("label") == "tfidf")
    assert tfidf_rows.get_column("runtime_seconds").to_list() == [1.5]
    assert tfidf_rows.get_column("similarity_backend").to_list() == ["sklearn"]
    assert tfidf_rows.get_column("is_exact_backend").to_list() == [True]
    assert tfidf_rows.get_column("country").to_list() == ["gb"]
    assert tfidf_rows.get_column("target_rows").to_list() == [2]
    # Tokenizer is its own axis, read straight off each
    # entry's own config -- both configs here use "wordpiece" (_build_config's
    # default), which is itself a case worth asserting since a `representation`
    # switch alone must not silently change it.
    assert tfidf_rows.get_column("tokenizer").to_list() == ["wordpiece:promoted"]
    wordpiece_rows = pruned.filter(pl.col("label") == "wordpiece")
    assert wordpiece_rows.get_column("tokenizer").to_list() == ["wordpiece:promoted"]

    # Same target, same country, same transform -> one population key
    # and one truth key shared by both representations, built with no refusal
    # -- a representation sweep over one target passes by default.
    population_keys = pruned.get_column("population_key").unique().to_list()
    assert population_keys == [population_keys[0]]
    assert population_keys[0] is not None
    truth_keys = pruned.get_column("truth_key").unique().to_list()
    assert truth_keys == [truth_keys[0]]
    assert truth_keys[0] is not None


def test_build_strategy_comparison_refuses_entries_with_different_population_keys(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Two entries whose `population_key` disagrees for a country they both
    scored cannot be pooled into one comparison by default -- they scored
    different populations, and a blended row would hide that."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)
    assert result.keys is not None
    mismatched = replace(
        result,
        keys=replace(result.keys, target_population="a-different-digest"),
    )

    with pytest.raises(ValueError, match="population_key"):
        build_strategy_comparison(
            [
                StrategyRunEntry(label="baseline", config=config, result=result),
                StrategyRunEntry(label="variant", config=config, result=mismatched),
            ]
        )


def test_build_strategy_comparison_refuses_entries_with_different_truth_keys(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """The same refusal applies to `truth_key`: two entries scored against
    different resolved truth maps for one country are not comparable either,
    even when they agree on population."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)
    assert result.keys is not None
    mismatched = replace(
        result,
        keys=replace(result.keys, truth="a-different-digest"),
    )

    with pytest.raises(ValueError, match="truth_key"):
        build_strategy_comparison(
            [
                StrategyRunEntry(label="baseline", config=config, result=result),
                StrategyRunEntry(label="variant", config=config, result=mismatched),
            ]
        )


def test_build_strategy_comparison_allow_mixed_population_bypasses_the_refusal(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """`allow_mixed_population=True` is the "caller asks for the mixed view"
    escape hatch: no refusal, and the disagreeing keys stay visible as
    columns."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)
    assert result.keys is not None
    mismatched = replace(
        result,
        keys=replace(result.keys, target_population="a-different-digest"),
    )

    comparison = build_strategy_comparison(
        [
            StrategyRunEntry(label="baseline", config=config, result=result),
            StrategyRunEntry(label="variant", config=config, result=mismatched),
        ],
        allow_mixed_population=True,
    )

    pruned = comparison.filter(
        (pl.col("stage") == "pruned") & (pl.col("population") == "universe")
    )
    by_label = dict(
        zip(
            pruned.get_column("label").to_list(),
            pruned.get_column("population_key").to_list(),
            strict=True,
        )
    )
    assert by_label["baseline"] != by_label["variant"]


def test_build_strategy_comparison_reports_recall_at_k_and_candidate_set_size_ratio(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, mocker
) -> None:
    """Both `recall_at_k` and `candidate_set_size_ratio` reach `strategy_comparison`.

    `candidate_set_size_ratio` is checkable against the fixture's own known
    2 (source) x 2 (target) exhaustive space -- `source_rows` is the *full*
    per-country source population (both fixture rows), not just the
    ground-truth-labelled subset `reduction_ratio` above is scoped to.
    """
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    mocker.patch("blocking.workflow._tokenize", side_effect=_fake_tokenize)

    result = execute_blocking_run(config)
    comparison = build_strategy_comparison(
        [StrategyRunEntry(label="tfidf", config=config, result=result)]
    )

    pruned = comparison.filter(pl.col("stage") == "pruned").row(0, named=True)
    assert pruned["source_rows"] == 2
    assert pruned["candidate_pair_count"] == result.matched_edges.height
    assert pruned["candidate_set_size_ratio"] == pytest.approx(
        result.matched_edges.height / (2 * 2)
    )
    assert 0.0 <= pruned["recall_at_k"] <= 1.0


@pytest.mark.integration
def test_recall_at_k_and_candidate_set_size_ratio_are_repeatable_across_runs(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, mocker
) -> None:
    """Candidate generation is deterministic, and that extends to these metrics.

    Two `execute_blocking_run()` calls against the identical config/fixture
    produce byte-identical `recall_at_k`/`candidate_set_size_ratio` (and the
    blended figures they sit beside), not merely values in a plausible
    range. Marked integration: a real (if tiny) end-to-end run, twice over.
    """
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    mocker.patch("blocking.workflow._tokenize", side_effect=_fake_tokenize)

    first_comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="tfidf", config=config, result=execute_blocking_run(config)
            )
        ]
    )
    second_comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="tfidf", config=config, result=execute_blocking_run(config)
            )
        ]
    )

    report = check_strategy_comparison_repeatability(
        first_comparison, second_comparison
    )
    assert report.get_column("matches").all()


def test_is_exact_backend_classifies_every_known_similarity_backend() -> None:
    # dense_brute is a chunked exhaustive scan, not an approximation, so it
    # classifies exact alongside sklearn/sparse_dot_topn -- see
    # `_EXACT_SIMILARITY_BACKENDS`'s comment for why.
    assert _is_exact_backend("sklearn") is True
    assert _is_exact_backend("sparse_dot_topn") is True
    assert _is_exact_backend("dense_brute") is True
    assert _is_exact_backend("svd_rerank") is False
    # hnsw is a genuine sub-linear ANN index (usearch), approximate by
    # construction -- its own recall against dense_brute is what a caller
    # prices, never gated.
    assert _is_exact_backend("hnsw") is False
    # kmeans/hdbscan route to one partition rather than scanning the whole
    # target, so neither fixed classification fits them.
    assert _is_exact_backend("kmeans") is None
    assert _is_exact_backend("hdbscan") is None


def test_build_strategy_comparison_marks_svd_rerank_as_inexact(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots, similarity_backend="svd_rerank")
    result = execute_blocking_run(config)

    comparison = build_strategy_comparison(
        [StrategyRunEntry(label="tfidf-svd", config=config, result=result)]
    )

    assert comparison.get_column("is_exact_backend").unique().to_list() == [False]
    assert comparison.get_column("runtime_seconds").null_count() == comparison.height


def test_build_strategy_comparison_without_ground_truth_keeps_run_with_null_metrics(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture_without_ground_truth(layer_fixture_dir)
    config = _build_config(workspace_roots, require_source_ground_truth=False)
    result = execute_blocking_run(config)
    assert result.pair_truth_eval is None

    comparison = build_strategy_comparison(
        [StrategyRunEntry(label="tfidf", config=config, result=result)]
    )

    assert comparison.height == 2  # one "pruned" row, one "raw" row -- both null-metric
    assert comparison.get_column("precision").null_count() == 2
    assert comparison.get_column("label").to_list() == ["tfidf", "tfidf"]
    assert set(comparison.get_column("stage").to_list()) == {"pruned", "raw"}


def test_accelerator_settings_separates_a_pruned_run_from_its_own_baseline(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    # The pooling hazard this column exists for: both runs are tfidf/sklearn
    # with the same label, so every other discriminating column is equal.
    _write_gleif_gb_fixture(layer_fixture_dir)
    accelerated_config = _build_config(
        workspace_roots,
        backend_options={"prefix_filter": True},
    )
    baseline_config = _build_config(workspace_roots)

    comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="tfidf",
                config=accelerated_config,
                result=execute_blocking_run(accelerated_config),
            ),
            StrategyRunEntry(
                label="tfidf",
                config=baseline_config,
                result=execute_blocking_run(baseline_config),
            ),
        ]
    )

    for column in ("label", "representation", "similarity_backend"):
        assert comparison.get_column(column).n_unique() == 1
    assert sorted(comparison.get_column("accelerator_settings").unique().to_list()) == [
        "exact_name_filter=true",
        "exact_name_filter=true;prefix_filter=true",
    ]


def test_comparison_rows_carry_the_runs_own_phase_figures(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Every row of a run carries its country's index build and scan
    wall-clock and the run's peak resident set size, read from the run's
    own phase records, so cost sits beside recall on one row."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)

    comparison = build_strategy_comparison(
        [StrategyRunEntry(label="tfidf", config=config, result=result)]
    )

    by_routine = {record["routine"]: record for record in result.timings}
    row = comparison.filter(pl.col("country") == "gb").row(0, named=True)
    assert row["target_index_seconds"] == by_routine["target_index"]["elapsed_seconds"]
    assert row["scoring_seconds"] == by_routine["scoring"]["elapsed_seconds"]
    rss_values = [
        rss
        for rss in (record["peak_rss_bytes"] for record in result.timings)
        if isinstance(rss, int)
    ]
    assert row["peak_rss_bytes"] == max(rss_values)
    assert row["runtime_seconds"] is None


def test_accelerator_settings_records_the_workflow_level_exact_match_fast_path(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots, exact_name_filter=False)

    comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="tfidf", config=config, result=execute_blocking_run(config)
            )
        ]
    )

    assert comparison.get_column("accelerator_settings").unique().to_list() == [
        "exact_name_filter=false"
    ]


def test_accelerator_settings_is_key_sorted_so_it_groups_deterministically(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    first = _build_config(
        workspace_roots, backend_options={"prefix_filter": True, "force": True}
    )
    second = _build_config(
        workspace_roots, backend_options={"force": True, "prefix_filter": True}
    )

    assert _encode_accelerator_settings(first.strategy) == _encode_accelerator_settings(
        second.strategy
    )


def test_accelerator_settings_rejects_a_colliding_backend_option(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, backend_options={"exact_name_filter": False}
    )

    with pytest.raises(ValueError, match="exact_name_filter"):
        _encode_accelerator_settings(config.strategy)


def test_accelerator_settings_is_populated_on_a_run_without_ground_truth(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    # The null-metric path builds its row from a dict rather than the eval
    # frame, so it is a separate place the column has to be filled in.
    _write_gleif_gb_fixture_without_ground_truth(layer_fixture_dir)
    config = _build_config(workspace_roots, require_source_ground_truth=False)

    comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="tfidf", config=config, result=execute_blocking_run(config)
            )
        ]
    )

    assert comparison.get_column("accelerator_settings").null_count() == 0


def test_build_strategy_comparison_carries_tokenizer_as_its_own_axis(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A tokenizer-candidate comparison is one more labelled run in
    this framework -- two entries differing only in `tokenizer`
    (same `representation`) must be distinguishable on their own column,
    not folded into `label`/`representation`.
    """
    _write_gleif_gb_fixture(layer_fixture_dir)
    wordpiece_config = _build_config(
        workspace_roots, representation="tfidf", tokenizer="wordpiece"
    )
    sentencepiece_config = _build_config(
        workspace_roots, representation="tfidf", tokenizer="sentencepiece"
    )
    # The same trainer under another stored tokenizer's label.
    naive_config = _build_config(
        workspace_roots,
        representation="tfidf",
        tokenizer="wordpiece",
        tokenizer_profile="naive",
    )

    comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="wordpiece-candidate",
                config=wordpiece_config,
                result=execute_blocking_run(wordpiece_config),
            ),
            StrategyRunEntry(
                label="sentencepiece-candidate",
                config=sentencepiece_config,
                result=execute_blocking_run(sentencepiece_config),
            ),
            StrategyRunEntry(
                label="naive-candidate",
                config=naive_config,
                result=execute_blocking_run(naive_config),
            ),
        ]
    )

    by_label = {
        row["label"]: row["tokenizer"]
        for row in comparison.filter(pl.col("stage") == "pruned").iter_rows(named=True)
    }
    assert by_label == {
        "wordpiece-candidate": "wordpiece:promoted",
        "sentencepiece-candidate": "sentencepiece_bpe:promoted",
        "naive-candidate": "wordpiece:naive",
    }


def test_build_strategy_comparison_carries_name_transform_as_its_own_axis(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Two runs differing only in what both sides' names passed through are
    a transform comparison, readable on their own column rather than folded
    into `label`.

    `identity` and `cleanse` genuinely score different text (their
    `population_key` differs), so this pair is exactly the "caller asks
    for the mixed view" case `allow_mixed_population` exists for -- the
    comparison is the point of the test, not an oversight."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    raw_config = _build_config(workspace_roots, name_transform="identity")
    cleansed_config = _build_config(workspace_roots, name_transform="cleanse")

    comparison = build_strategy_comparison(
        [
            StrategyRunEntry(
                label="raw", config=raw_config, result=execute_blocking_run(raw_config)
            ),
            StrategyRunEntry(
                label="cleansed",
                config=cleansed_config,
                result=execute_blocking_run(cleansed_config),
            ),
        ],
        allow_mixed_population=True,
    )

    by_label = {
        row["label"]: row["name_transform"]
        for row in comparison.filter(pl.col("stage") == "pruned").iter_rows(named=True)
    }
    assert by_label == {
        "raw": "identity:default;preprocess:default",
        "cleansed": "cleanse:default;preprocess:default",
    }


def test_build_strategy_comparison_rejects_empty_entries() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_strategy_comparison([])


def test_write_strategy_comparison_report_writes_parquet(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
    layer_fixture_dir,
    comparison_location,
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)
    comparison = build_strategy_comparison(
        [StrategyRunEntry(label="tfidf", config=config, result=result)]
    )

    output_dir = tmp_path / "comparison_out"
    path = write_strategy_comparison_report(comparison_location(output_dir), comparison)

    assert path == output_dir / "strategy_comparison.parquet"
    assert path.exists()
    reread = pl.read_parquet(path)
    assert reread.columns == comparison.columns


def _write_gleif_gb_fixture_with_cleansed_different_residual(layer_dir_for) -> None:
    # gleif:1/gb:1: byte-identical raw name -- the fast path, so it
    # never enters the residual. gleif:2/gb:2: still different after
    # cleansing -- the one never-equal pair, exercising a non-null
    # name_equality_never_recall through build_strategy_comparison().
    matched_dir = _write_partition(
        layer_dir_for,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "acme ltd",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
            {
                "system_uri": "gleif:2",
                "name": "beta holdings limited",
                "name_cleansed": "beta holdings",
                "jurisdiction_code": "gb",
                "match_uri": "gb:2",
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
            {
                "system_uri": "gb:2",
                "name": "beta holdings group",
                "name_cleansed": "beta holdings group",
                "jurisdiction_code": "gb",
            },
        ],
    )


def test_build_strategy_comparison_carries_every_population(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Every population of `pair_truth_eval` reaches `strategy_comparison` as
    its own row, so a comparison reads the `never` residual's matrix directly
    rather than a blended figure beside it."""
    _write_gleif_gb_fixture_with_cleansed_different_residual(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)
    assert result.pair_truth_eval is not None

    comparison = build_strategy_comparison(
        [StrategyRunEntry(label="tfidf", config=config, result=result)]
    )

    pruned = comparison.filter(pl.col("stage") == "pruned")
    rows = {row["population"]: row for row in pruned.iter_rows(named=True)}
    assert set(rows) == {
        "universe",
        "raw",
        "basic",
        "cleansed",
        "preprocessed",
        "never",
        "unknown",
    }
    assert pruned.height == len(rows)
    # The universe and the residual are scored over different sources here --
    # the fast-path pair counts toward the universe's truth_pairs but not the
    # residual's, which is the whole point of reading the residual instead.
    assert rows["universe"]["truth_pairs"] == 2
    assert rows["never"]["truth_pairs"] == 1
    assert rows["never"]["tp"] == 1
    assert rows["never"]["recall"] == pytest.approx(1.0)


def _detail_row(
    *,
    source_id: str,
    target_id: str,
    is_truth_pair: bool,
    found: bool | None,
    similarity: float | None,
    rank: int | None,
    bucket: str = "never",
) -> dict[str, object]:
    return {
        "source_system": "gleif",
        "target_system": "gb",
        "country": "gb",
        "source_id": source_id,
        "target_id": target_id,
        "source_name": source_id,
        "source_name_cleansed": source_id,
        "target_name": target_id,
        "target_name_cleansed": target_id,
        "name_equality": bucket,
        "is_truth_pair": is_truth_pair,
        "found": found,
        "similarity": similarity,
        "rank": rank,
    }


def test_diff_pair_recovery_names_the_pair_the_variant_found_and_baseline_missed() -> (
    None
):
    """The recovery attribution names the truth
    pair the second run found and the first missed, with its similarity --
    not a delta between the two runs' aggregate metrics.
    """
    baseline_detail = pl.DataFrame(
        [
            _detail_row(
                source_id="gleif:1",
                target_id="gb:1",
                is_truth_pair=True,
                found=False,
                similarity=None,
                rank=None,
            ),
            _detail_row(
                source_id="gleif:2",
                target_id="gb:2",
                is_truth_pair=True,
                found=True,
                similarity=0.95,
                rank=1,
            ),
        ]
    )
    variant_detail = pl.DataFrame(
        [
            _detail_row(
                source_id="gleif:1",
                target_id="gb:1",
                is_truth_pair=True,
                found=True,
                similarity=0.42,
                rank=3,
            ),
            _detail_row(
                source_id="gleif:2",
                target_id="gb:2",
                is_truth_pair=True,
                found=True,
                similarity=0.95,
                rank=1,
            ),
        ]
    )

    recovery = diff_pair_recovery(baseline_detail, variant_detail)

    assert recovery.height == 1
    row = recovery.row(0, named=True)
    assert row["source_id"] == "gleif:1"
    assert row["target_id"] == "gb:1"
    assert row["variant_similarity"] == pytest.approx(0.42)
    assert row["variant_rank"] == 3


def test_diff_pair_recovery_ignores_predicted_but_untrue_pairs() -> None:
    """A predicted-but-untrue row (`is_truth_pair=False`, `found=None`) has
    no "recovered" reading and must never surface as a false recovery."""
    baseline_detail = pl.DataFrame(
        [
            _detail_row(
                source_id="gleif:1",
                target_id="gb:1",
                is_truth_pair=False,
                found=None,
                similarity=0.3,
                rank=2,
            )
        ]
    )
    variant_detail = pl.DataFrame(
        [
            _detail_row(
                source_id="gleif:1",
                target_id="gb:1",
                is_truth_pair=False,
                found=None,
                similarity=0.6,
                rank=1,
            )
        ]
    )

    recovery = diff_pair_recovery(baseline_detail, variant_detail)

    assert recovery.height == 0


def test_diff_pair_recovery_skips_a_pair_found_by_neither_or_both_runs() -> None:
    baseline_detail = pl.DataFrame(
        [
            _detail_row(
                source_id="gleif:1",
                target_id="gb:1",
                is_truth_pair=True,
                found=False,
                similarity=None,
                rank=None,
            ),
            _detail_row(
                source_id="gleif:2",
                target_id="gb:2",
                is_truth_pair=True,
                found=True,
                similarity=0.9,
                rank=1,
            ),
        ]
    )
    variant_detail = pl.DataFrame(
        [
            _detail_row(
                source_id="gleif:1",
                target_id="gb:1",
                is_truth_pair=True,
                found=False,
                similarity=None,
                rank=None,
            ),
            _detail_row(
                source_id="gleif:2",
                target_id="gb:2",
                is_truth_pair=True,
                found=True,
                similarity=0.9,
                rank=1,
            ),
        ]
    )

    recovery = diff_pair_recovery(baseline_detail, variant_detail)

    assert recovery.height == 0
