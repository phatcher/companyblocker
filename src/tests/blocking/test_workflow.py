import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from company_vectorize.clustering_contract import TfidfTargetIndexBuildSettings

from blocking import workflow as workflow_module
from blocking.comparison import (
    StrategyRunEntry,
    build_strategy_comparison,
    diff_pair_recovery,
)
from blocking.contracts import (
    BlockingRunConfig,
    BlockingStrategyConfig,
    EmptySourceLoadError,
    validate_blocking_run_config,
)
from blocking.loader import load_dataset_descriptor
from blocking.reporting import (
    build_blocking_summary,
    write_blocking_report,
    write_run_manifest,
)
from blocking.truth import ColumnTruth
from blocking.workflow import (
    _CASEFOLD_COL,
    _TOKEN_COL,
    _apply_similarity_ratio_pruning,
    _apply_target_cap,
    _build_diagnostic_frame,
    _build_source_truth_map,
    _compute_similarity_distribution,
    _index_part_key,
    _resolve_tokenizer_path,
    _split_exact_name_matches,
    _tokenize,
    compute_recall_curve_for_run,
    execute_blocking_run,
    execute_name_variant_cross_system_run,
    execute_name_variant_recovery_run,
    remeasure_pair_truth_eval,
    resolve_blocking_countries,
    resolve_blocking_run_keys,
)
from tests.promoted_tokenizers import promoted_tokenizer_files
from validation.public_benchmark_materializer import (
    PublicBenchmarkMaterializationConfig,
    materialize_public_benchmark,
)
from validation.runner import pair_truth_eval_row
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    layer_directory,
    perturbed_dataset_dir,
)
from workspace.identity import RunKeys
from workspace.layer_layout import layer_partition_dir
from workspace.pointer import NothingPromotedError
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


def _build_config(
    roots: WorkspaceRoots,
    *,
    representation: str = "tfidf",
    text_view: str | None = None,
    tokenizer: str = "wordpiece",
    max_candidates_per_target: int | None = None,
    candidate_similarity_ratio: float | None = None,
    similarity_backend: str = "sklearn",
    backend_options: dict[str, object] | None = None,
) -> BlockingRunConfig:
    source = load_dataset_descriptor(roots=roots, system="gleif")
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
        text_view=text_view,
        tokenizer=tokenizer,
        max_candidates_per_target=max_candidates_per_target,
        candidate_similarity_ratio=candidate_similarity_ratio,
    )
    return BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
    )


def _write_names_sidecar(
    layer_dir_for,
    *,
    system: str,
    date: str,
    filename: str,
    rows: list[dict[str, object]],
) -> None:
    sidecar_dir = layer_dir_for(system, layer=CANONICAL_LAYER_NAME) / date
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(sidecar_dir / filename)


def _event_count(event: dict[str, object], key: str) -> int:
    """Read a numeric field off a progress payload.

    `ProgressCallback` is `Callable[[dict[str, object]], None]`, so every field
    arrives as `object`; asserting the count fields really are ints is part of
    what the progress contract promises.
    """
    value = event[key]
    assert isinstance(value, int)
    return value


# -- Direct tests of workflow.py's own standalone helpers ------------------
#
# Each of these imports the helper by its own path and asserts its
# contract in isolation, independent of `execute_blocking_run`'s wiring.


def test_resolve_blocking_countries_intersects_then_filters() -> None:
    from blocking.contracts import BlockingDatasetDescriptor

    def _descriptor(countries: tuple[str, ...]) -> BlockingDatasetDescriptor:
        return BlockingDatasetDescriptor(
            system="x",
            system_dir=Path("/data/x/matched"),
            layer="matched",
            has_ground_truth=True,
            matched_target_systems=(),
            available_countries=countries,
        )

    source = _descriptor(("gb", "fr", "ie"))
    target = _descriptor(("fr", "ie", "de"))

    assert resolve_blocking_countries(
        source=source, target=target, configured_countries=None
    ) == ["fr", "ie"]
    assert resolve_blocking_countries(
        source=source, target=target, configured_countries=("ie",)
    ) == ["ie"]
    assert (
        resolve_blocking_countries(
            source=source, target=target, configured_countries=("de",)
        )
        == []
    )


def test_build_source_truth_map_entity_row_reads_cached_column(
    workspace_roots: WorkspaceRoots,
) -> None:
    # An entity-scheme row (scheme == system) is answered straight off the
    # already-loaded frame's own match_uri column -- no matched/ layer needs
    # to exist on disk at all for this branch, proving it never re-reads it.
    source_frame = pl.DataFrame(
        {
            "system_uri": ["gleif://1", "gleif://2"],
            "match_uri": ["gb://1", None],
        }
    )

    result = _build_source_truth_map(
        source_frame, roots=workspace_roots, system="gleif"
    )

    assert result.sort("source_id").to_dicts() == [
        {"source_id": "gleif://1", "source_match_uri": "gb://1"},
        {"source_id": "gleif://2", "source_match_uri": None},
    ]


def test_build_source_truth_map_derived_scheme_row_walks_the_uri_chain(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    # A row whose own system_uri is a derived scheme (name-variant here)
    # cannot be answered from the source frame's own match_uri column, since
    # that column only ever reflects *this* row's own (absent) matched/
    # entry -- it must be walked back to the entity it stands in for.
    matched_dir = layer_fixture_dir("gleif", layer=MATCHED_LAYER_NAME)
    matched_dir.mkdir(parents=True)
    pl.DataFrame({"system_uri": ["gleif://1"], "match_uri": ["gb://1"]}).write_parquet(
        matched_dir / "gleif-001.parquet"
    )
    names_dir = (
        layer_fixture_dir("gleif", layer=CANONICAL_LAYER_NAME) / "2026-01-01" / "names"
    )
    names_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "system_uri": ["name://gleif/1/0000000000000abc"],
            "source_uri": ["gleif://1"],
            "name": ["Acme"],
        }
    ).write_parquet(names_dir / "gleif-names-001.parquet")

    source_frame = pl.DataFrame({"system_uri": ["name://gleif/1/0000000000000abc"]})

    result = _build_source_truth_map(
        source_frame, roots=workspace_roots, system="gleif"
    )

    assert result.to_dicts() == [
        {"source_id": "name://gleif/1/0000000000000abc", "source_match_uri": "gb://1"}
    ]


def test_build_source_truth_map_empty_frame_returns_empty_schema(
    workspace_roots: WorkspaceRoots,
) -> None:
    result = _build_source_truth_map(
        pl.DataFrame(schema={"system_uri": pl.Utf8}),
        roots=workspace_roots,
        system="gleif",
    )

    assert result.height == 0
    assert result.schema == {"source_id": pl.Utf8, "source_match_uri": pl.Utf8}


def test_tokenize_hands_the_tokenizer_a_lowercased_copy_and_leaves_name_alone(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every tokenizer here was trained on lowercased text and carries no
    normalizer, so a capital letter is an unknown token: the tokenizer must
    see lowercase, while `name` itself keeps its case for the raw
    name-equality level and the exact-name fast path."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, representation="wordpiece", text_view="tokens"
    )
    seen: dict[str, list[str]] = {}

    def fake_tokenize(frame: pl.DataFrame, **kwargs: object) -> pl.DataFrame:
        name_col = str(kwargs["name_col"])
        seen["texts"] = frame.get_column(name_col).to_list()
        return frame.with_columns(
            pl.col(name_col).str.split(" ").alias(str(kwargs["token_col"]))
        )

    monkeypatch.setattr(workflow_module, "tokenize_name_dataframe", fake_tokenize)
    frame = pl.DataFrame({"system_uri": ["gb:1"], "name": ["Acme HOLDINGS Ltd"]})
    promoted_tokenizer_files(workspace_roots, system="gb")

    tokenized = _tokenize(frame, config)

    assert seen["texts"] == ["acme holdings ltd"]
    assert tokenized.get_column(_TOKEN_COL).to_list() == [["acme", "holdings", "ltd"]]
    assert tokenized.get_column("name").to_list() == ["Acme HOLDINGS Ltd"]
    assert _CASEFOLD_COL not in tokenized.columns


def test_tokenize_reads_the_target_tokenizer_for_source_rows(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source tokens split by the source system's own vocabulary are not
    comparable with the target's, so a `gleif` frame in a run against `gb` is
    tokenized with `gb`'s tokenizer, the only one promoted here."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, representation="wordpiece", text_view="tokens"
    )
    seen: dict[str, object] = {}

    def fake_tokenize(frame: pl.DataFrame, **kwargs: object) -> pl.DataFrame:
        seen["tokenizer_path"] = kwargs["tokenizer_path"]
        return frame.with_columns(
            pl.col(str(kwargs["name_col"]))
            .str.split(" ")
            .alias(str(kwargs["token_col"]))
        )

    monkeypatch.setattr(workflow_module, "tokenize_name_dataframe", fake_tokenize)
    gb_model = promoted_tokenizer_files(workspace_roots, system="gb").model
    frame = pl.DataFrame({"system_uri": ["gleif:1"], "name": ["Acme Holdings"]})

    _tokenize(frame, config)

    assert seen["tokenizer_path"] == gb_model


def test_the_tokenizer_profile_picks_which_stored_tokenizer_the_run_reads(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """The run names a label and the pointer says which tokenizer that is:
    `promoted` unless the strategy says otherwise, and a label that names
    nothing for the target is refused by name."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, representation="wordpiece", text_view="tokens"
    )
    tuned = promoted_tokenizer_files(workspace_roots, system="gb", key="tuned").model
    naive = promoted_tokenizer_files(
        workspace_roots, system="gb", key="first_guess", profile="naive"
    ).model

    def _path(profile: str) -> Path:
        strategy = replace(config.strategy, tokenizer_profile=profile)
        return _resolve_tokenizer_path(replace(config, strategy=strategy))

    assert _resolve_tokenizer_path(config) == tuned
    assert _path("naive") == naive and naive != tuned
    with pytest.raises(NothingPromotedError, match="no untried reference"):
        _path("untried")


def test_the_tokenizer_encoding_picks_which_sentencepiece_tokenizer_the_run_reads(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """`bpe` and `unigram` are promoted separately, so a SentencePiece run
    reads the one its encoding names, `bpe` unless the strategy says
    otherwise."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, representation="sentencepiece", text_view="tokens"
    )
    config = replace(
        config, strategy=replace(config.strategy, tokenizer="sentencepiece")
    )
    bpe, unigram = (
        promoted_tokenizer_files(
            workspace_roots,
            system="gb",
            trainer="sentencepiece",
            tokenizer_encoding=encoding,
            key=encoding,
        ).model
        for encoding in ("bpe", "unigram")
    )

    unigram_config = replace(
        config, strategy=replace(config.strategy, tokenizer_encoding="unigram")
    )

    assert _resolve_tokenizer_path(config) == bpe
    assert _resolve_tokenizer_path(unigram_config) == unigram and unigram != bpe


def test_a_tokenizer_path_is_used_as_it_is_and_no_profile_is_looked_up(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, tmp_path: Path
) -> None:
    """A candidate no profile names is reached by its file, with nothing
    promoted at all, and a path naming no file is refused."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, representation="wordpiece", text_view="tokens"
    )
    candidate = tmp_path / "models" / "v21000_mf8.json"
    candidate.parent.mkdir()
    candidate.write_text("{}", encoding="utf-8")

    def _with(path: Path) -> BlockingRunConfig:
        return replace(
            config, strategy=replace(config.strategy, tokenizer_path=str(path))
        )

    assert _resolve_tokenizer_path(_with(candidate)) == candidate
    with pytest.raises(FileNotFoundError, match="names no file"):
        _resolve_tokenizer_path(_with(tmp_path / "absent.json"))


def test_index_part_key_moves_with_representation_noise_and_tokenizer_content(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Every setting that changes what gets embedded moves the index key: the
    representation, the noise-word resource, and the target tokenizer's
    content, which a promotion overwrites in place under the same scope,
    profile and trainer."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    settings = TfidfTargetIndexBuildSettings(ngram_min=1, ngram_max=2)
    tokenizer = promoted_tokenizer_files(workspace_roots, system="gb").model
    tokenizer.write_text("gb-vocab", encoding="utf-8")

    def _key(config: BlockingRunConfig, *, text_view: str) -> str:
        return _index_part_key(
            config,
            target_population_key="pop-key",
            effective_text_view=text_view,
            build_settings=settings,
        )

    tfidf = _build_config(workspace_roots, representation="tfidf")
    base = _key(tfidf, text_view="text")
    assert base == _key(tfidf, text_view="text")
    assert base != _key(
        replace(
            tfidf,
            strategy=replace(
                tfidf.strategy, preprocess_profile="default|+noise_words:balanced"
            ),
        ),
        text_view="text",
    )
    assert base != _key(
        replace(tfidf, strategy=replace(tfidf.strategy, representation="wordpiece")),
        text_view="text",
    )

    wordpiece = _build_config(
        workspace_roots, representation="wordpiece", text_view="tokens"
    )
    before = _key(wordpiece, text_view="tokens")
    tokenizer.write_text("gb-vocab-optimized", encoding="utf-8")
    assert _key(wordpiece, text_view="tokens") != before


def test_apply_similarity_ratio_pruning_keeps_only_near_best_per_source() -> None:
    edges = pl.DataFrame(
        {
            "source_id": ["s1", "s1", "s1", "s2"],
            "target_id": ["t1", "t2", "t3", "t4"],
            "similarity": [0.9, 0.85, 0.5, 0.6],
        }
    )

    pruned = _apply_similarity_ratio_pruning(edges, ratio=0.8)

    assert sorted(pruned.get_column("target_id").to_list()) == ["t1", "t2", "t4"]


def test_apply_similarity_ratio_pruning_noop_on_empty_frame() -> None:
    edges = pl.DataFrame(schema={"source_id": pl.Utf8, "similarity": pl.Float64})

    assert _apply_similarity_ratio_pruning(edges, ratio=0.8).height == 0


def test_apply_target_cap_keeps_top_n_per_target() -> None:
    edges = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3"],
            "target_id": ["t1", "t1", "t1"],
            "similarity": [0.9, 0.8, 0.7],
        }
    )

    pruned = _apply_target_cap(edges, max_per_target=2)

    assert sorted(pruned.get_column("source_id").to_list()) == ["s1", "s2"]


def test_apply_target_cap_noop_on_empty_frame() -> None:
    edges = pl.DataFrame(
        schema={"source_id": pl.Utf8, "target_id": pl.Utf8, "similarity": pl.Float64}
    )

    assert _apply_target_cap(edges, max_per_target=1).height == 0


def test_compute_similarity_distribution_buckets_known_values() -> None:
    edges = pl.DataFrame({"similarity": [0.0, 0.5, 1.0]})

    distribution = _compute_similarity_distribution(edges, min_similarity=0.0)

    assert distribution.height == 20
    assert distribution.get_column("edge_count").sum() == 3
    by_bucket = distribution.get_column("edge_count").to_list()
    assert by_bucket[0] == 1
    assert by_bucket[10] == 1
    assert by_bucket[19] == 1
    assert distribution.row(0, named=True)["bucket_start"] == pytest.approx(0.0)
    assert distribution.row(19, named=True)["bucket_end"] == pytest.approx(1.0)


def test_compute_similarity_distribution_empty_input_is_zero_filled() -> None:
    edges = pl.DataFrame(schema={"similarity": pl.Float64})

    distribution = _compute_similarity_distribution(edges, min_similarity=0.2)

    assert distribution.height == 20
    assert distribution.get_column("edge_count").sum() == 0
    assert distribution.row(0, named=True)["bucket_start"] == pytest.approx(0.2)
    assert distribution.row(19, named=True)["bucket_end"] == pytest.approx(1.0)


def test_split_exact_name_matches_resolves_byte_identical_names() -> None:
    source_frame = pl.DataFrame(
        {
            "system_uri": ["s1", "s2"],
            "name": ["acme limited", "beta trading"],
        }
    )
    target_frame = pl.DataFrame(
        {
            "system_uri": ["t1", "t2"],
            "name": ["acme limited", "beta trdng ltd"],
        }
    )

    residual, exact_edges = _split_exact_name_matches(
        source_frame, target_frame, country="gb"
    )

    assert residual.get_column("system_uri").to_list() == ["s2"]
    assert exact_edges.to_dicts() == [
        {
            "source_id": "s1",
            "target_id": "t1",
            "similarity": 1.0,
            "rank": 1,
            "country": "gb",
        }
    ]


def test_split_exact_name_matches_ranks_multiple_targets_by_target_id() -> None:
    source_frame = pl.DataFrame({"system_uri": ["s1"], "name": ["acme limited"]})
    target_frame = pl.DataFrame(
        {
            "system_uri": ["t2", "t1"],
            "name": ["acme limited", "acme limited"],
        }
    )

    residual, exact_edges = _split_exact_name_matches(
        source_frame, target_frame, country="gb"
    )

    assert residual.height == 0
    exact_edges = exact_edges.sort("rank")
    assert exact_edges.get_column("target_id").to_list() == ["t1", "t2"]
    assert exact_edges.get_column("rank").to_list() == [1, 2]
    assert exact_edges.get_column("similarity").to_list() == [1.0, 1.0]


def test_split_exact_name_matches_null_name_never_matches_null_name() -> None:
    source_frame = pl.DataFrame(
        {"system_uri": ["s1"], "name": pl.Series([None], dtype=pl.Utf8)}
    )
    target_frame = pl.DataFrame(
        {"system_uri": ["t1"], "name": pl.Series([None], dtype=pl.Utf8)}
    )

    residual, exact_edges = _split_exact_name_matches(
        source_frame, target_frame, country="gb"
    )

    assert exact_edges.height == 0
    assert residual.get_column("system_uri").to_list() == ["s1"]


def test_split_exact_name_matches_no_matches_is_noop() -> None:
    source_frame = pl.DataFrame({"system_uri": ["s1"], "name": ["beta trading"]})
    target_frame = pl.DataFrame({"system_uri": ["t1"], "name": ["beta trdng ltd"]})

    residual, exact_edges = _split_exact_name_matches(
        source_frame, target_frame, country="gb"
    )

    assert residual.equals(source_frame)
    assert exact_edges.height == 0


def test_build_diagnostic_frame_includes_name_cleansed_when_present() -> None:
    """`name_cleansed` is carried through when `text_frame` has it and it
    differs from `input_name_col` -- the canonicalization step this
    diagnostics snapshot exists to make visible."""
    text_frame = pl.DataFrame(
        {
            "system_uri": ["s1"],
            "name": ["Acme Ltd."],
            "name_cleansed": ["acme ltd"],
            "cluster_text": ["acme ltd"],
        }
    )

    frame = _build_diagnostic_frame(
        text_frame,
        id_col="system_uri",
        id_alias="source_id",
        country="gb",
        input_name_col="name",
        feature_col="cluster_text",
    )

    assert frame.to_dicts() == [
        {
            "source_id": "s1",
            "country": "gb",
            "name": "Acme Ltd.",
            "name_cleansed": "acme ltd",
            "cluster_text": "acme ltd",
        }
    ]


def test_build_diagnostic_frame_omits_name_cleansed_when_absent() -> None:
    """No `name_cleansed` column in `text_frame` is skipped rather than
    erroring -- not every caller runs the canonicalization step."""
    text_frame = pl.DataFrame(
        {
            "system_uri": ["t1"],
            "name": ["acme ltd"],
            "cluster_text": ["acme ltd"],
        }
    )

    frame = _build_diagnostic_frame(
        text_frame,
        id_col="system_uri",
        id_alias="target_id",
        country="gb",
        input_name_col="name",
        feature_col="cluster_text",
    )

    assert "name_cleansed" not in frame.columns
    assert frame.to_dicts() == [
        {
            "target_id": "t1",
            "country": "gb",
            "name": "acme ltd",
            "cluster_text": "acme ltd",
        }
    ]


# -- `execute_blocking_run`'s own decision logic ----------------------------
#
# These are the only entry point onto orchestration-level branches that live
# inside `execute_blocking_run` itself (no extractable helper exists for
# them). Each is narrow, fast (small synthetic fixtures, no real clustering),
# and untagged -- calling the real entry point is not on its own grounds for
# `integration`; breadth and cost both have to be
# present.


def test_execute_blocking_run_emits_progress_events(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots, representation="tfidf")

    events: list[dict[str, object]] = []
    execute_blocking_run(config, source_chunk_size=1, progress_callback=events.append)

    phases = [event["phase"] for event in events]
    assert "country_start" in phases
    assert "country_progress" in phases
    assert "country_complete" in phases

    start_event = next(event for event in events if event["phase"] == "country_start")
    assert start_event["country"] == "gb"
    assert _event_count(start_event, "source_rows") >= 1
    assert _event_count(start_event, "target_rows") >= 1

    complete_event = next(
        event for event in events if event["phase"] == "country_complete"
    )
    assert _event_count(complete_event, "matched_edges") >= 1

    progress_events = [
        event for event in events if event["phase"] == "country_progress"
    ]
    assert len(progress_events) >= 1
    for event in progress_events:
        assert _event_count(event, "rows_scored") <= _event_count(event, "source_rows")


def test_name_forms_are_derived_once_and_read_by_every_later_run_over_the_same_names(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, mocker
) -> None:
    """The forms depend on the names and the two profiles and on nothing else
    a strategy sets, so a second run, under another backend, reads them from
    the name forms store; a run under another profile derives its own."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    derive = mocker.spy(workflow_module, "derive_name_forms")

    first = execute_blocking_run(_build_config(workspace_roots), source_chunk_size=1)
    assert derive.call_count == 2

    derive.reset_mock()
    other_backend = _build_config(workspace_roots, similarity_backend="lsh")
    second = execute_blocking_run(other_backend, source_chunk_size=1)
    assert derive.call_count == 0
    assert second.exact_match_summary.equals(first.exact_match_summary)

    noisy = _build_config(workspace_roots)
    noisy = replace(
        noisy,
        strategy=replace(
            noisy.strategy, preprocess_profile="default|+noise_words:strict"
        ),
    )
    execute_blocking_run(noisy, source_chunk_size=1)
    assert derive.call_count == 2


def test_the_vectorizer_reads_the_preprocessed_name_and_the_scored_name_is_left_alone(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, mocker
) -> None:
    """One step decides what text is vectorized: the cleansed name with its
    company type removed, and its noise words too when the run asks. The
    scored column keeps the company type, for the joins and the diagnostics."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    seen: list[pl.DataFrame] = []
    original = workflow_module.build_clustering_text_view

    def spy(frame: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        assert kwargs["name_col"] == "name_preprocessed"
        seen.append(frame.select("name_cleansed", "name_preprocessed"))
        return original(frame, **kwargs)

    mocker.patch.object(workflow_module, "build_clustering_text_view", side_effect=spy)
    config = _build_config(workspace_roots, representation="tfidf")

    execute_blocking_run(config, source_chunk_size=1)

    rows = pl.concat(seen).unique().rows()
    assert rows
    assert all(cleansed.startswith(processed) for cleansed, processed in rows)
    assert any(cleansed != processed for cleansed, processed in rows)
    assert not any(processed.split()[-1] in {"ltd", "limited"} for _, processed in rows)


def test_a_run_completes_with_the_lsh_backend_on_a_sparse_representation(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """`lsh` scores by estimated Jaccard similarity over minhash signatures,
    so the fixture's identical names are what it is sure to pair."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, representation="tfidf", similarity_backend="lsh"
    )

    result = execute_blocking_run(config, source_chunk_size=1)

    assert result.matched_edges.height >= 1


def test_the_lsh_backend_is_refused_for_a_dense_representation(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(
        workspace_roots, representation="sbert", similarity_backend="lsh"
    )

    with pytest.raises(ValueError, match="'lsh'.*dense representation 'sbert'"):
        validate_blocking_run_config(config)


def test_execute_blocking_run_reuses_target_index_cache(
    workspace_roots: WorkspaceRoots, layer_fixture_dir, mocker
) -> None:
    """The blocking workflow resolves its target index through the same
    shared cache `src/validation`'s own harness uses, so a second run
    against the same target under the same strategy never rebuilds it."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots, representation="tfidf")

    from company_vectorize import resolve_clustering_strategy as resolve_strategy

    original_strategy = resolve_strategy("tfidf")
    build_target_index_spy = mocker.spy(original_strategy, "build_target_index")
    mocker.patch.object(
        workflow_module,
        "resolve_clustering_strategy",
        return_value=original_strategy,
    )

    first = execute_blocking_run(config, source_chunk_size=1)
    assert first.matched_edges.height >= 1
    assert build_target_index_spy.call_count == 1

    build_target_index_spy.reset_mock()
    second = execute_blocking_run(config, source_chunk_size=1)
    assert second.matched_edges.height >= 1
    assert build_target_index_spy.call_count == 0
    assert second.keys == first.keys


def test_execute_blocking_run_announces_and_records_each_phase(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Every pre-scoring phase is announced on entry and exit through the
    progress seam, carrying the rows it works on and its cost on exit, and
    the same phases land on the result as timings."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots, representation="tfidf")

    events: list[dict[str, object]] = []
    result = execute_blocking_run(config, progress_callback=events.append)

    started = [event["name"] for event in events if event["phase"] == "phase_start"]
    completed = [event for event in events if event["phase"] == "phase_complete"]
    assert started == [
        "load_frames",
        "population_keys",
        "name_forms",
        "exact_name_split",
        "text_view",
        "target_index",
        "backend_index",
        "scoring",
        "pruning",
        "truth_eval",
    ]
    # The scoring phase also reports its two steps, each summed over the
    # chunks, once it has ended.
    recorded = [
        *started[: started.index("scoring") + 1],
        "scoring_vectorize_source",
        "scoring_backend",
        *started[started.index("scoring") + 1 :],
    ]
    assert [event["name"] for event in completed] == recorded
    index_event = next(event for event in completed if event["name"] == "target_index")
    assert index_event["country"] == "gb"
    assert _event_count(index_event, "rows") >= 1
    elapsed = index_event["elapsed_seconds"]
    assert isinstance(elapsed, float) and elapsed >= 0.0
    rss = index_event["rss_bytes"]
    assert rss is None or (isinstance(rss, int) and rss > 0)

    assert [record["routine"] for record in result.timings] == recorded
    assert all(record["label"] == "gb" for record in result.timings)
    scoring = next(
        record for record in result.timings if record["routine"] == "scoring"
    )
    steps = [
        record
        for record in result.timings
        if record["routine"] in ("scoring_vectorize_source", "scoring_backend")
    ]
    step_seconds = [
        seconds
        for record in steps
        if isinstance(seconds := record["elapsed_seconds"], float)
    ]
    assert len(step_seconds) == len(steps)
    assert all(seconds >= 0.0 for seconds in step_seconds)
    scoring_seconds = scoring["elapsed_seconds"]
    assert isinstance(scoring_seconds, float)
    assert sum(step_seconds) <= scoring_seconds
    rows_in = scoring["rows_in"]
    assert isinstance(rows_in, int) and rows_in >= 1
    rows_out = scoring["rows_out"]
    assert isinstance(rows_out, int)
    assert (
        rows_out >= result.raw_matched_edges.filter(pl.col("similarity") < 1.0).height
    )


def test_exact_join_cascade_resolves_each_form_and_scans_only_the_rest(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """One truth pair at each name-equality level plus an unlabelled row:
    the raw, basic and cleansed pairs get their form's exact edge and are
    counted under it, every source is still scanned, a pair the scan also
    scored appears once, and the evaluation finds every matched level in
    full."""
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:raw",
                "name": "acme ltd",
                "jurisdiction_code": "gb",
                "match_uri": "gb:raw",
            },
            {
                "system_uri": "gleif:basic",
                "name": "Beta Holdings",
                "jurisdiction_code": "gb",
                "match_uri": "gb:basic",
            },
            {
                "system_uri": "gleif:cleansed",
                "name": "gamma limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:cleansed",
            },
            {
                "system_uri": "gleif:never",
                "name": "delta logistics",
                "jurisdiction_code": "gb",
                "match_uri": "gb:never",
            },
            {
                "system_uri": "gleif:unlabelled",
                "name": "unrelated co",
                "jurisdiction_code": "gb",
                "match_uri": None,
            },
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb:raw", "name": "acme ltd", "jurisdiction_code": "gb"},
            # The renamed twin of gb:raw: an identical name elsewhere must not
            # stop the source reaching it through the scan.
            {
                "system_uri": "gb:twin",
                "name": "acme limited",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:basic",
                "name": "beta holdings",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:cleansed",
                "name": "gamma ltd",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:never",
                "name": "delta logistic services",
                "jurisdiction_code": "gb",
            },
        ],
    )
    config = _build_config(workspace_roots)

    events: list[dict[str, object]] = []
    result = execute_blocking_run(config, progress_callback=events.append)

    summary = result.exact_match_summary.row(0, named=True)
    assert summary["exact_match_count"] == 3
    assert (
        summary["resolved_by_raw"],
        summary["resolved_by_basic"],
        summary["resolved_by_cleansed"],
        summary["resolved_by_transform"],
    ) == (1, 1, 1, 0)
    assert summary["multi_target_source_count"] == 0
    assert summary["unmatched_source_row_count"] == 2

    scoring = next(
        record for record in result.timings if record["routine"] == "scoring"
    )
    # Every source is scanned, the matched ones included.
    assert scoring["rows_in"] == 5
    start_event = next(event for event in events if event["phase"] == "country_start")
    assert start_event["source_rows"] == 5

    exact_edges = result.raw_matched_edges.filter(pl.col("similarity") == 1.0)
    assert set(exact_edges.get_column("source_id").to_list()) == {
        "gleif:raw",
        "gleif:basic",
        "gleif:cleansed",
    }
    # The scan scores the identical pair too; it appears once, as the exact
    # edge, and the source still reaches its renamed twin.
    pairs = result.raw_matched_edges.select("source_id", "target_id")
    assert pairs.height == pairs.unique().height
    raw_source_targets = set(
        result.raw_matched_edges.filter(pl.col("source_id") == "gleif:raw")
        .get_column("target_id")
        .to_list()
    )
    assert {"gb:raw", "gb:twin"} <= raw_source_targets
    assert result.pair_truth_eval is not None
    for level in ("raw", "basic", "cleansed"):
        row = pair_truth_eval_row(result.pair_truth_eval, level)
        assert (row["truth_pairs"], row["recall"]) == (1, 1.0), level
    assert pair_truth_eval_row(result.pair_truth_eval, "never")["truth_pairs"] == 1


def test_execute_blocking_run_target_cap_prunes_and_costs_recall(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:0",
                "name": "acme limited",
                "jurisdiction_code": "gb",
                "match_uri": None,
            },
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
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )
    config = _build_config(
        workspace_roots, representation="tfidf", max_candidates_per_target=1
    )

    result = execute_blocking_run(config)

    # gleif:0 and gleif:1 share identical text, so their similarity to gb:1 is
    # tied; the target cap's source_id tie-break keeps gleif:0 (sorts first)
    # and drops gleif:1 -- the actual true match.
    assert result.raw_matched_edges.height == 2
    assert result.matched_edges.height == 1
    assert result.matched_edges.get_column("source_id").to_list() == ["gleif:0"]

    assert result.pruning_summary.height == 1
    summary_row = result.pruning_summary.row(0, named=True)
    assert summary_row["raw_edge_count"] == 2
    assert summary_row["dropped_by_target_cap"] == 1
    assert summary_row["pruned_edge_count"] == 1

    assert result.raw_pair_truth_eval is not None
    assert result.pair_truth_eval is not None
    raw_row = result.raw_pair_truth_eval.row(0, named=True)
    pruned_row = result.pair_truth_eval.row(0, named=True)
    assert raw_row["tp"] == 1
    assert raw_row["recall"] == 1.0
    assert pruned_row["tp"] == 0
    assert pruned_row["fn"] == 1
    assert pruned_row["recall"] == 0.0


def test_execute_blocking_run_empty_matched_edges_gives_empty_clusters(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "   ",
                "jurisdiction_code": "gb",
                "match_uri": None,
            }
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )
    config = _build_config(workspace_roots, representation="tfidf")

    result = execute_blocking_run(config)

    assert result.clusters.height == 0
    assert set(result.clusters.columns) == {
        "cluster_id",
        "node_id",
        "node_name",
        "node_role",
    }


def test_execute_blocking_run_fails_when_source_resolves_to_zero_rows(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A country with a real, present partition directory that happens to
    hold zero rows -- the shape a read racing a concurrent writer's swap
    actually produces, per `read_country_partition_frame`'s own "absent
    directory and empty one round-trip the same" docstring -- fails the run
    rather than being absorbed as a legitimately empty candidate set.
    """

    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "acme limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            }
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    # Overwrite with a same-schema, zero-row file: the partition directory
    # and its parquet file both exist (so this run's country is still
    # discovered/scored), but the load itself resolves nothing.
    pl.DataFrame(
        schema={
            "system_uri": pl.Utf8,
            "name": pl.Utf8,
            "jurisdiction_code": pl.Utf8,
            "match_uri": pl.Utf8,
        }
    ).write_parquet(layer_partition_dir(matched_dir, value="gb") / "part-00001.parquet")
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )
    config = _build_config(workspace_roots, representation="tfidf")

    with pytest.raises(EmptySourceLoadError, match="gleif.*gb"):
        execute_blocking_run(config)


def test_diff_pair_recovery_attributes_a_pair_a_stricter_run_missed(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Two labelled runs of the *same* canonical pairing, differing
    only in `min_similarity`, are diffed by their own per-pair outcomes --
    `diff_pair_recovery()` names the truth pair the looser run recovered,
    rather than a delta between the two runs' aggregate precision/recall.
    """
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "acme holdings",
                "name_cleansed": "acme holdings",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            }
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {
                "system_uri": "gb:1",
                "name": "acme group",
                "name_cleansed": "acme group",
                "jurisdiction_code": "gb",
            }
        ],
    )
    baseline_config = _build_config(
        workspace_roots, representation="tfidf", similarity_backend="sklearn"
    )

    strict_strategy = replace(baseline_config.strategy, min_similarity=0.99)
    strict_config = replace(baseline_config, strategy=strict_strategy)
    baseline = execute_blocking_run(strict_config)

    loose_strategy = replace(baseline_config.strategy, min_similarity=0.01)
    loose_config = replace(baseline_config, strategy=loose_strategy)
    variant = execute_blocking_run(loose_config)

    # The canonical baseline's own metrics are unaffected by this
    # item -- both runs above are ordinary execute_blocking_run() calls,
    # never a variant-expanded index folded into one shared run.
    assert baseline.pair_truth_eval is not None
    assert baseline.pair_truth_eval.row(0, named=True)["tp"] == 0
    assert variant.pair_truth_eval is not None
    assert variant.pair_truth_eval.row(0, named=True)["tp"] == 1

    assert baseline.pair_truth_eval_detail is not None
    assert variant.pair_truth_eval_detail is not None
    recovery = diff_pair_recovery(
        baseline.pair_truth_eval_detail, variant.pair_truth_eval_detail
    )
    assert recovery.height == 1
    row = recovery.row(0, named=True)
    assert row["source_id"] == "gleif:1"
    assert row["target_id"] == "gb:1"
    assert row["variant_similarity"] is not None
    assert row["variant_similarity"] > 0.0

    # Reuse comparison.build_strategy_comparison() for the aggregate
    # side rather than a second reshaper.
    aggregate = build_strategy_comparison(
        [
            StrategyRunEntry(label="strict", config=strict_config, result=baseline),
            StrategyRunEntry(label="loose", config=loose_config, result=variant),
        ]
    )
    assert set(aggregate.get_column("label").to_list()) == {"strict", "loose"}


def test_execute_name_variant_recovery_run_self_referential_ground_truth(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A system's own recorded name variant scored against that same
    system's own primary records -- ground truth is the variant row's own
    `source_uri` back-reference, never a cross-system `matched/` walk (`gb`
    here has no `matched/` layer at all).
    """
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"},
            {"system_uri": "gb:2", "name": "omega plc", "jurisdiction_code": "gb"},
        ],
    )
    variant_frame = pl.DataFrame(
        {
            "system_uri": ["name://gleif/1/0000000000abc123"],
            "source_uri": ["gb:1"],
            "name": ["acme limited"],
            "name_type": ["previous"],
        }
    )
    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.01,
        max_candidates_per_source=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )

    result = execute_name_variant_recovery_run(
        descriptor,
        variant_frame=variant_frame,
        strategy=strategy,
        roots=workspace_roots,
    )

    assert result.pair_truth_eval is not None
    truth_row = result.pair_truth_eval.row(0, named=True)
    assert truth_row["tp"] == 1
    assert (
        "name://gleif/1/0000000000abc123"
        in result.matched_edges.get_column("source_id").to_list()
    )
    assert "gb:1" in result.matched_edges.get_column("target_id").to_list()
    assert result.keys is not None


def test_resolve_blocking_run_keys_matches_the_keys_the_run_records(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """The key pass reads what the run would consume, so the keys it gives
    before scoring are the keys scoring records."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)

    keys = resolve_blocking_run_keys(config)

    assert keys.truth is not None
    assert execute_blocking_run(config, expected_keys=keys).keys == keys


def test_execute_blocking_run_refuses_expected_keys_that_no_longer_match(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A layer rewritten between the key pass and scoring would file outputs
    under keys that no longer describe them, so the run raises instead."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    keys = resolve_blocking_run_keys(config)

    with pytest.raises(ValueError, match="inputs changed"):
        execute_blocking_run(config, expected_keys=replace(keys, index="stale"))


def test_resolve_blocking_run_keys_survive_rewriting_a_layer_with_identical_rows(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Rewriting a layer with the same consumed rows, here with an unrelated
    column added, leaves every key unchanged under every transform: no key
    reads a file's bytes or a materialized column, only `system_uri` and the
    forms the run derives live from `name`.
    """
    _write_gleif_gb_fixture(layer_fixture_dir)

    for name_transform in ("identity", "cleanse"):
        config = _build_config(workspace_roots)
        config = replace(
            config, strategy=replace(config.strategy, name_transform=name_transform)
        )
        before = resolve_blocking_run_keys(config)

        _write_partition(
            layer_fixture_dir,
            system="gb",
            layer="canonical",
            country="gb",
            rows=[
                {
                    "system_uri": "gb:1",
                    "name": "acme ltd",
                    "jurisdiction_code": "gb",
                    "extra_unrelated_column": f"rewritten-{name_transform}",
                },
                {
                    "system_uri": "gb:2",
                    "name": "omega plc",
                    "jurisdiction_code": "gb",
                    "extra_unrelated_column": f"rewritten-{name_transform}",
                },
            ],
        )

        assert resolve_blocking_run_keys(config) == before, name_transform


def test_resolve_blocking_run_keys_a_consumed_row_moves_population_and_index_only(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Changing a target name the run scores moves the target population key
    and the index key built on it, and nothing else."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    before = resolve_blocking_run_keys(config)

    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb:1", "name": "acme limited", "jurisdiction_code": "gb"},
            {"system_uri": "gb:2", "name": "omega plc", "jurisdiction_code": "gb"},
        ],
    )
    after = resolve_blocking_run_keys(config)

    assert after.target_population != before.target_population
    assert after.index != before.index
    assert after.source_population == before.source_population
    assert after.truth == before.truth
    assert after.settings == before.settings


def test_resolve_blocking_run_keys_a_truth_row_moves_only_the_truth_key(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Changing a source row's recorded match, and not its name, moves only
    the truth key: the rows scored are the same, only what they are checked
    against changed."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    before = resolve_blocking_run_keys(config)

    _write_partition(
        layer_fixture_dir,
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
                "match_uri": "gb:2",
            },
        ],
    )
    after = resolve_blocking_run_keys(config)

    assert after.truth != before.truth
    assert replace(after, truth=before.truth) == before


def test_resolve_blocking_run_keys_a_cleanse_profile_moves_settings_and_index_not_population(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """The population key covers the raw names read, so a cleanse profile
    change leaves it where it is and moves the settings key and the index key,
    whose embedded text the profile shapes, under every transform.

    Mixed-case raw names, unlike `_write_gleif_gb_fixture`'s already-lowercase
    fixture, so `default|-lowercase` (skip the profile's lowercase stage)
    changes the derived text.
    """
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "Acme Limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb:1", "name": "ACME LTD", "jurisdiction_code": "gb"},
        ],
    )

    def _keys(*, name_transform: str, cleanse_profile: str) -> RunKeys:
        config = _build_config(workspace_roots)
        config = replace(
            config,
            strategy=replace(
                config.strategy,
                name_transform=name_transform,
                cleanse_profile=cleanse_profile,
            ),
        )
        return resolve_blocking_run_keys(config)

    for name_transform in ("identity", "cleanse"):
        default = _keys(name_transform=name_transform, cleanse_profile="default")
        alternate = _keys(
            name_transform=name_transform, cleanse_profile="default|-lowercase"
        )

        assert alternate.source_population == default.source_population
        assert alternate.target_population == default.target_population
        assert alternate.settings != default.settings
        assert alternate.index != default.index


def test_execute_blocking_run_cleanse_transform_compares_both_sides_cleansed(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Under the cleanse transform both sides are cleansed live and compared
    on the derived name: a pair whose raw spellings differ resolves on the
    exact-name fast path, the name-equality lookups read the live forms, and
    the run records what it applied. No `name_cleansed` exists in the fixture
    for the run to have read from disk."""
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "Acme Limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb:1", "name": "ACME LTD", "jurisdiction_code": "gb"},
            {"system_uri": "gb:2", "name": "Omega Plc", "jurisdiction_code": "gb"},
        ],
    )
    config = _build_config(workspace_roots)
    config = replace(
        config, strategy=replace(config.strategy, name_transform="cleanse")
    )

    result = execute_blocking_run(config)

    assert result.name_transform == {
        "kind": "cleanse",
        "column": "name_cleansed",
        "cleanse_profile": "default",
        "preprocess_profile": "default",
    }
    assert result.exact_match_summary.row(0, named=True)["exact_match_count"] == 1
    assert result.pair_truth_eval is not None
    assert pair_truth_eval_row(result.pair_truth_eval, "universe")["tp"] == 1
    # Raw and basic forms differ ("acme limited" against "acme ltd"); the
    # full cleanse makes them equal, so the pair sits at the cleansed level,
    # classified from the forms this run derived.
    assert pair_truth_eval_row(result.pair_truth_eval, "raw")["truth_pairs"] == 0
    cleansed = pair_truth_eval_row(result.pair_truth_eval, "cleansed")
    assert cleansed["truth_pairs"] == 1
    assert cleansed["tp"] == 1
    summary = build_blocking_summary(result)
    assert summary["name_transform"] == result.name_transform


def test_execute_blocking_run_never_reads_the_layers_materialized_name_cleansed(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Every run derives both sides' forms live, whichever column it scores.
    Both layers here carry a `name_cleansed` that says the pair differs;
    the live cleanse says it is equal, and the run believes the live one:
    under `identity` the raw names differ, the live cleansed forms join, so
    the cascade resolves the pair at the `cleansed` level with no scan, where
    the stale materialised forms would have left it to a scan that misses."""
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "Acme Limited",
                "name_cleansed": "stale source form",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {
                "system_uri": "gb:1",
                "name": "ACME LTD",
                "name_cleansed": "stale target form",
                "jurisdiction_code": "gb",
            },
        ],
    )
    config = _build_config(workspace_roots)
    config = replace(
        config,
        strategy=replace(
            config.strategy, min_similarity=0.99, name_transform="identity"
        ),
    )

    result = execute_blocking_run(config)

    assert result.name_transform["kind"] == "identity"
    assert result.exact_match_summary.row(0, named=True)["resolved_by_cleansed"] == 1
    assert result.pair_truth_eval is not None
    assert pair_truth_eval_row(result.pair_truth_eval, "universe")["tp"] == 1
    assert pair_truth_eval_row(result.pair_truth_eval, "cleansed")["truth_pairs"] == 1
    assert pair_truth_eval_row(result.pair_truth_eval, "never")["truth_pairs"] == 0
    assert result.pair_truth_eval_detail is not None
    detail = result.pair_truth_eval_detail.row(0, named=True)
    assert detail["source_name_cleansed"] == "acme ltd"
    assert detail["target_name_cleansed"] == "acme ltd"


def test_execute_blocking_run_scores_a_perturbed_dataset_against_its_own_system(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A materialized perturbed set scored against the system it was made
    from: the source is a perturbed selector, the target is that system's
    own cleansed layer, and each row's truth is its `source_uri`, read as a
    column. No matched/ layer exists anywhere in the fixture."""
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
            "system_uri": ["perturbed://ie/p1/1/a", "perturbed://ie/p1/1/b"],
            "source_uri": ["ie://1", "ie://2"],
            "match_uri": [None, None],
            "name": ["acme limted", "zeta corp"],
            "jurisdiction_code": ["ie", "ie"],
        }
    ).write_parquet(partition_dir / "part-00000.parquet")
    _write_partition(
        layer_fixture_dir,
        system="ie",
        layer="canonical",
        country="ie",
        rows=[
            {"system_uri": "ie://1", "name": "acme limited", "jurisdiction_code": "ie"},
            {"system_uri": "ie://2", "name": "omega plc", "jurisdiction_code": "ie"},
        ],
    )
    truth = ColumnTruth("source_uri")
    source = load_dataset_descriptor(
        roots=workspace_roots, system="perturbed://ie/p1/v1/1", truth=truth
    )
    target = load_dataset_descriptor(
        roots=workspace_roots, system="ie", require_ground_truth=False
    )
    config = BlockingRunConfig(
        roots=workspace_roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=BlockingStrategyConfig(
            representation="tfidf",
            top_k=2,
            min_similarity=0.2,
            max_candidates_per_source=None,
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
            # Whole words: under character n-grams two unrelated names share
            # enough letters to clear the low threshold this fixture uses.
            tfidf_analyzer="word",
        ),
        truth=truth,
    )

    result = execute_blocking_run(config)

    assert result.pair_truth_eval is not None
    truth_row = result.pair_truth_eval.row(0, named=True)
    # Both perturbed rows are truth pairs; the typo'd one is recovered, the
    # renamed one is not.
    assert truth_row["truth_pairs"] == 2
    assert truth_row["tp"] == 1
    assert truth_row["fn"] == 1
    edges = result.matched_edges.to_dicts()
    assert any(
        edge["source_id"] == "perturbed://ie/p1/1/a" and edge["target_id"] == "ie://1"
        for edge in edges
    )
    summary = build_blocking_summary(result)
    assert summary["source_system"] == "perturbed://ie/p1/v1/1"
    assert summary["target_system"] == "ie"


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason=(
        "The public benchmark materializer still writes each table as a system's "
        "cleansed layer, which no run reads: a run reads canonical and matched. "
        "It scores again once the benchmark is a production a run is handed by "
        "reference, and this marker then fails the suite until it is removed."
    ),
)
def test_execute_blocking_run_scores_a_materialized_public_benchmark(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
) -> None:
    """A public two-table benchmark, materialized through
    `public_benchmark_materializer` from a `tableA.csv`/`tableB.csv`/
    `matches.csv` fixture and scored: the queried side (table B) is the
    source, its `source_uri` is the truth column, the same
    perturbed-dataset pattern the test above exercises, and the
    doubly-matched record's two extra rows both recover as their own truth
    pair. Guards the adapter without a real Zenodo download. What the
    materializer itself writes is asserted in
    `src/tests/validation/test_public_benchmark_materializer.py`."""
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    pl.DataFrame(
        [
            {"id": 0, "name": "Acme Widget", "description": "a fine widget"},
            {"id": 1, "name": "Zenith Gadget", "description": "a gadget"},
        ]
    ).write_csv(benchmark_root / "tableA.csv")
    pl.DataFrame(
        [
            {"id": 0, "name": "Acme Widgets Inc", "description": "widget maker"},
            {"id": 1, "name": "Other Corp", "description": "unrelated"},
            {"id": 2, "name": "Zenith Gadgets", "description": "gadget maker"},
        ]
    ).write_csv(benchmark_root / "tableB.csv")
    pl.DataFrame(
        [
            {"ltable_id": 0, "rtable_id": 0},
            {"ltable_id": 1, "rtable_id": 2},
            {"ltable_id": 0, "rtable_id": 2},
        ]
    ).write_csv(benchmark_root / "matches.csv")

    materialized = materialize_public_benchmark(
        PublicBenchmarkMaterializationConfig(
            roots=workspace_roots,
            benchmark_root=benchmark_root,
            benchmark_name="testbench",
        )
    )
    assert materialized.indexed_system == "testbench-a"
    assert materialized.queried_system == "testbench-b"
    assert materialized.truth_pairs == 3
    assert materialized.multi_match_source_count == 1

    truth = ColumnTruth("source_uri")
    source = load_dataset_descriptor(
        roots=workspace_roots, system=materialized.queried_system, truth=truth
    )
    target = load_dataset_descriptor(
        roots=workspace_roots,
        system=materialized.indexed_system,
        require_ground_truth=False,
    )
    config = BlockingRunConfig(
        roots=workspace_roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=BlockingStrategyConfig(
            representation="tfidf",
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
            tfidf_analyzer="word",
        ),
        truth=truth,
    )

    result = execute_blocking_run(config)

    assert result.pair_truth_eval is not None
    universe = pair_truth_eval_row(result.pair_truth_eval, "universe")
    assert universe["truth_pairs"] == 3
    assert universe["tp"] == 3
    summary = build_blocking_summary(result)
    assert summary["source_system"] == "testbench-b"
    assert summary["target_system"] == "testbench-a"


def test_execute_name_variant_recovery_run_requires_source_uri(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A sidecar whose rows still only share `system_uri` many-to-one with
    their primary record, carrying no per-row `source_uri` identity, cannot
    be scored as their own source records -- caught explicitly, not
    silently scoring nothing.
    """
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )
    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.01,
        max_candidates_per_source=None,
    )
    variant_frame = pl.DataFrame(
        {"system_uri": ["gb:1"], "name": ["acme limited"], "name_type": ["previous"]}
    )

    with pytest.raises(ValueError, match="source_uri"):
        execute_name_variant_recovery_run(
            descriptor,
            variant_frame=variant_frame,
            strategy=strategy,
            roots=workspace_roots,
        )


def test_execute_name_variant_cross_system_run_ground_truth_from_matched_layer(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A source system's own recorded name variant scored against a
    *different* target system's primary records -- ground truth is not a
    local back-reference (gb never appears as `descriptor`'s own primary
    population here) but `gleif`'s own recorded cross-system match, reached
    by walking the `name://` row back to the `gleif` primary record that
    carries it and reading that record's own `match_uri`.
    """
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif://1",
                "name": "acme corp",
                "jurisdiction_code": "gb",
                "match_uri": "gb://1",
            }
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb://1", "name": "acme ltd", "jurisdiction_code": "gb"},
            {"system_uri": "gb://2", "name": "omega plc", "jurisdiction_code": "gb"},
        ],
    )
    _write_names_sidecar(
        layer_fixture_dir,
        system="gleif",
        date="2026-01-01",
        filename="gleif-names-001.parquet",
        rows=[
            {
                "system_uri": "name://gleif/1/0000000000abc123",
                "source_uri": "gleif://1",
                "name": "acme limited",
                "name_type": "previous",
            }
        ],
    )
    descriptor = load_dataset_descriptor(roots=workspace_roots, system="gleif")
    target_descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.01,
        max_candidates_per_source=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )

    result = execute_name_variant_cross_system_run(
        descriptor, target_descriptor, strategy=strategy, roots=workspace_roots
    )

    assert result.pair_truth_eval is not None
    truth_row = result.pair_truth_eval.row(0, named=True)
    assert truth_row["tp"] == 1
    assert (
        "name://gleif/1/0000000000abc123"
        in result.matched_edges.get_column("source_id").to_list()
    )
    assert "gb://1" in result.matched_edges.get_column("target_id").to_list()
    assert result.keys is not None


def test_execute_name_variant_cross_system_run_requires_matched_layer(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """`descriptor` resolved without a `matched` layer has no
    `data/<system>/matched/` for the cross-system walk to terminate into --
    caught explicitly rather than silently scoring every row as untrue.
    """
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gleif://1", "name": "acme corp", "jurisdiction_code": "gb"}
        ],
    )
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb://1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )
    _write_names_sidecar(
        layer_fixture_dir,
        system="gleif",
        date="2026-01-01",
        filename="gleif-names-001.parquet",
        rows=[
            {
                "system_uri": "name://gleif/1/0000000000abc123",
                "source_uri": "gleif://1",
                "name": "acme limited",
                "name_type": "previous",
            }
        ],
    )
    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gleif", require_ground_truth=False
    )
    target_descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.01,
        max_candidates_per_source=None,
    )

    with pytest.raises(ValueError, match="matched"):
        execute_name_variant_cross_system_run(
            descriptor, target_descriptor, strategy=strategy, roots=workspace_roots
        )


def test_execute_name_variant_cross_system_run_requires_source_uri(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Same guard as `execute_name_variant_recovery_run()`: a sidecar whose
    rows still only share `system_uri` many-to-one with their primary
    record cannot be scored as their own source records.
    """
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif://1",
                "name": "acme corp",
                "jurisdiction_code": "gb",
                "match_uri": "gb://1",
            }
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb://1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )
    descriptor = load_dataset_descriptor(roots=workspace_roots, system="gleif")
    target_descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.01,
        max_candidates_per_source=None,
    )
    variant_frame = pl.DataFrame(
        {
            "system_uri": ["gleif://1"],
            "name": ["acme limited"],
            "name_type": ["previous"],
        }
    )

    with pytest.raises(ValueError, match="source_uri"):
        execute_name_variant_cross_system_run(
            descriptor,
            target_descriptor,
            variant_frame=variant_frame,
            strategy=strategy,
            roots=workspace_roots,
        )


# -- The one end-to-end path for the blocking stage -------------------------
#
# Everything above is narrow and cheap. This is the file's sole broad+slow
# path: real, unmocked sklearn clustering end to end through
# `execute_blocking_run()` (measured at 3 to 15s against a baseline
# under 0.3s, a tenfold gap rather than a gradient), so it is the one test
# tagged `integration` rather than run on every fast iteration.


@pytest.mark.integration
def test_execute_blocking_run_kmeans_partition_backend(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    # kmeans works with the existing sparse tfidf representation
    # unchanged -- no fake/mocked strategy needed, unlike the hdbscan test
    # this file no longer carries. With only 2 target rows, the default
    # n_clusters clamps down to 2 (partition_similarity.py's own clamp:
    # max(1, min(n_clusters, n_targets))), so routing degenerates to
    # nearest-target-row, matching the plain sklearn backend's own result for
    # this fixture.
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots, representation="tfidf")
    config = replace(
        config, strategy=replace(config.strategy, similarity_backend="kmeans")
    )

    result = execute_blocking_run(config)

    edges = result.matched_edges
    top_edges = edges.filter(pl.col("source_id") == "gleif:1")
    assert top_edges.height >= 1
    assert "gb:1" in top_edges.get_column("target_id").to_list()
    assert result.pair_truth_eval is not None
    assert result.pair_truth_eval.row(0, named=True)["tp"] == 1


# -- Containment, not equality, between the two renamed populations -


def _write_gleif_gb_fixture_with_name_cleansed_and_unlabelled(layer_dir_for) -> None:
    """Three source rows: one resolved by the exact-name fast path
    (`gleif:1`, byte-identical raw names), one needing the scored path but
    still labelled (`gleif:2`, cleansed-equal only), and one carrying no
    ground truth at all (`gleif:3`, `match_uri=None`) -- so the full source
    population (3) is strictly larger than the labelled population
    `pair_truth_eval_detail` can ever cover (2), making the containment this
    test asserts non-trivial rather than accidentally an equality.
    """

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
                "name": "beta holdings",
                "name_cleansed": "beta group",
                "jurisdiction_code": "gb",
                "match_uri": "gb:2",
            },
            {
                "system_uri": "gleif:3",
                "name": "unrelated co",
                "name_cleansed": "unrelated co",
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
            {
                "system_uri": "gb:1",
                "name": "acme ltd",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:2",
                "name": "beta group",
                "name_cleansed": "beta group",
                "jurisdiction_code": "gb",
            },
        ],
    )


def test_pair_truth_eval_detail_source_ids_contained_in_exact_match_summary_population(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """`pair_truth_eval_detail` covers every *labelled* source row, which is
    never more than the *full*
    source population `exact_match_summary` splits into
    `exact_match_count` + `unmatched_source_row_count`. Containment, not
    equality, is the checkable relation: `gleif:3` is in the full population
    (it counts toward one of `exact_match_summary`'s two figures) but is
    absent from `pair_truth_eval_detail` entirely, since it carries no
    ground truth to evaluate.
    """

    _write_gleif_gb_fixture_with_name_cleansed_and_unlabelled(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))

    assert result.pair_truth_eval_detail is not None
    detail_source_ids = set(
        result.pair_truth_eval_detail.get_column("source_id").to_list()
    )

    summary_row = result.exact_match_summary.row(0, named=True)
    full_population_size = (
        summary_row["exact_match_count"] + summary_row["unmatched_source_row_count"]
    )

    assert full_population_size == 3
    assert detail_source_ids == {"gleif:1", "gleif:2"}
    assert len(detail_source_ids) < full_population_size
    assert len(detail_source_ids) <= full_population_size


# -- Direct tests of remeasure_pair_truth_eval() ----------------------------


def test_remeasure_pair_truth_eval_rewrites_truth_and_leaves_candidates_untouched(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    """A re-measure against the *same* real matched/cleansed data the
    original run scored must reproduce the same pair_truth_eval/
    pair_truth_eval_detail numbers (the walk's cached fast path answers the
    same question the original column read did) while every candidate
    artefact -- read back from disk, not recomputed -- is untouched byte for
    byte, and pair_truth_eval.parquet/summary.json are the two files that
    actually change on disk.
    """

    _write_gleif_gb_fixture_with_name_cleansed_and_unlabelled(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)
    location = run_location(tmp_path / "run")
    run_dir = location.directory
    write_blocking_report(location, result)

    candidate_artefact_names = [
        "matched_edges.parquet",
        "raw_matched_edges.parquet",
        "clusters.parquet",
    ]
    bytes_before = {
        name: (run_dir / name).read_bytes() for name in candidate_artefact_names
    }
    countries_before = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )["countries"]
    pair_truth_eval_detail_before = pl.read_parquet(
        run_dir / "pair_truth_eval_detail.parquet"
    )

    # top_k=2 matches _build_config's own strategy.top_k, so recall_at_k
    # reproduces exactly what the original run scored (see
    # remeasure_pair_truth_eval's own docstring for why top_k isn't
    # inferred from the run directory automatically).
    remeasure_pair_truth_eval(location, roots=workspace_roots, top_k=2)

    for name in candidate_artefact_names:
        assert (run_dir / name).read_bytes() == bytes_before[name]

    summary_after = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_after["countries"] == countries_before

    pair_truth_eval_detail_after = pl.read_parquet(
        run_dir / "pair_truth_eval_detail.parquet"
    )
    # Row order isn't part of this frame's contract (the walk's per-country
    # source-id iteration order need not match the original scoring run's),
    # so compare sorted -- the values themselves must still match exactly.
    sort_cols = ["source_id", "target_id"]
    assert pair_truth_eval_detail_after.sort(sort_cols).equals(
        pair_truth_eval_detail_before.sort(sort_cols)
    )

    assert summary_after["truth_remeasured"]["remeasured"] is True
    assert summary_after["truth_remeasured"]["remeasured_at"]
    # Re-measured against the same matched and cleansed data, the truth key
    # reproduces exactly what the original run recorded.
    assert result.keys is not None
    assert summary_after["keys"]["truth_key"] == result.keys.truth


def test_remeasure_pair_truth_eval_truth_key_moves_with_a_regenerated_matched_layer(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    """A re-measure against a *changed* matched layer writes a different
    truth key -- what this function's own re-resolve is meant to keep
    honest, since only the truth changed and nothing else did."""
    _write_gleif_gb_fixture(layer_fixture_dir)
    config = _build_config(workspace_roots)
    result = execute_blocking_run(config)
    location = run_location(tmp_path / "run")
    run_dir = location.directory
    write_blocking_report(location, result)

    # gleif:2's match_uri moves from unmatched to matched against gb:2 --
    # the matched layer regenerated with new match decisions, candidate
    # artefacts on disk untouched.
    _write_partition(
        layer_fixture_dir,
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
                "match_uri": "gb:2",
            },
        ],
    )

    remeasure_pair_truth_eval(location, roots=workspace_roots, top_k=2)

    summary_after = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert result.keys is not None
    assert summary_after["keys"]["truth_key"] != result.keys.truth


def test_remeasure_pair_truth_eval_missing_pair_truth_eval_raises(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    # A run made with require_ground_truth=False never writes
    # pair_truth_eval.parquet at all -- nothing to re-measure.
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"}],
    )
    location = run_location(tmp_path / "run")
    location.directory.mkdir()

    with pytest.raises(FileNotFoundError):
        remeasure_pair_truth_eval(location, roots=workspace_roots)


# -- Direct tests of compute_recall_curve_for_run() -------------------------


def _write_recall_curve_manifest(location, *, roots: WorkspaceRoots) -> None:
    """A `manifest.json` naming the scale settings `_build_config()`'s
    strategy carries (`top_k=2`, `min_similarity=0.2`,
    `max_candidates_per_source=None`, `similarity_backend="sklearn"`), in
    the same `name -> {"value", "source", "surface"}` shape a real run's
    resolved configuration takes."""
    write_run_manifest(
        location,
        identity={"source_system": "gleif", "target_system": "gb"},
        commit=None,
        configuration={
            "similarity_backend": {
                "value": "sklearn",
                "source": "given",
                "surface": "--similarity-backend",
            },
            "min_similarity": {
                "value": 0.2,
                "source": "given",
                "surface": "--min-similarity",
            },
            "top_k": {"value": 2, "source": "given", "surface": "--top-k"},
            "max_candidates_per_source": {
                "value": None,
                "source": "default",
                "surface": "--max-candidates-per-source",
            },
        },
        roots=roots,
    )


def test_compute_recall_curve_for_run_reads_run_directory_without_rerun(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    """No candidate generation happens here: the curve and its summary are
    read entirely from `matched_edges.parquet`, `pair_truth_eval_detail.
    parquet`, `pair_truth_eval.parquet` and `manifest.json`, all already
    written by `write_blocking_report()`/`write_run_manifest()`."""

    _write_gleif_gb_fixture_with_name_cleansed_and_unlabelled(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))
    location = run_location(tmp_path / "run")
    run_dir = location.directory
    write_blocking_report(location, result)
    _write_recall_curve_manifest(location, roots=workspace_roots)

    report = compute_recall_curve_for_run(location)

    assert (run_dir / "recall_curve.parquet").exists()
    assert (run_dir / "recall_curve_summary.parquet").exists()
    assert report.curve.height > 0

    summary_row = report.summary.row(0, named=True)
    assert summary_row["similarity_backend"] == "sklearn"
    assert summary_row["is_exact_backend"] is True
    assert summary_row["min_similarity"] == 0.2
    assert summary_row["top_k"] == 2
    assert summary_row["max_candidates_per_source"] is None
    # gb's target layer carries 2 rows in this fixture.
    assert summary_row["target_rows"] == 2

    curve_on_disk = pl.read_parquet(run_dir / "recall_curve.parquet")
    assert curve_on_disk.equals(report.curve)


def test_compute_recall_curve_for_run_missing_manifest_raises(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, run_location
) -> None:
    # write_blocking_report() alone never writes manifest.json -- only
    # run_blocking.py's own write_run_manifest() call does.
    _write_gleif_gb_fixture_with_name_cleansed_and_unlabelled(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))
    location = run_location(tmp_path / "run")
    write_blocking_report(location, result)

    with pytest.raises(FileNotFoundError):
        compute_recall_curve_for_run(location)
