"""Direct tests of `blocking.audit`'s own verdict logic, plus one fixture run
through the real pipeline whose truth pairs an independent, fully
unrestricted rescore of the whole population -- the fixture-scale stand-in
for a real brute-force exact scan -- is checked against."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from blocking import audit
from blocking.comparison import StrategyRunEntry
from blocking.contracts import (
    BLOCKING_ARTIFACT_SCHEMAS,
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingRunResult,
    BlockingStrategyConfig,
)
from blocking.loader import load_dataset_descriptor
from blocking.reporting import produce_blocking_run, write_blocking_report
from blocking.run_layout import read_run_location, resolve_run_location_for
from blocking.workflow import (
    execute_blocking_run,
    rescore_missed_pairs,
    resolve_blocking_run_keys,
)
from validation.runner import PAIR_TRUTH_EVAL_DETAIL_COLUMNS
from workspace.records import read_record
from workspace.reference import reference_at
from workspace.roots import WorkspaceRoots, default_workspace_roots

from .test_workflow import _write_match_metadata, _write_partition

_ROOTS = default_workspace_roots(Path("/root"))
_MATCHED_EDGES_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["matched_edges"]
_EMPTY_MATCHED_EDGES = pl.DataFrame(schema=_MATCHED_EDGES_SCHEMA)
_EMPTY_DETAIL = pl.DataFrame(schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS)
_NO_FRAME = pl.DataFrame()


def _descriptor(system: str) -> BlockingDatasetDescriptor:
    return BlockingDatasetDescriptor(
        system=system,
        system_dir=Path(system),
        layer="matched" if system == "gleif" else "canonical",
        has_ground_truth=system == "gleif",
        matched_target_systems=("gb",) if system == "gleif" else (),
        available_countries=("gb",),
    )


def _config(strategy: BlockingStrategyConfig) -> BlockingRunConfig:
    return BlockingRunConfig(
        roots=_ROOTS,
        prepared_base_dir=None,
        source=_descriptor("gleif"),
        target=_descriptor("gb"),
        countries=("gb",),
        strategy=strategy,
    )


def _result(
    *,
    pair_truth_eval_detail: pl.DataFrame,
    raw_matched_edges: pl.DataFrame = _EMPTY_MATCHED_EDGES,
) -> BlockingRunResult:
    return BlockingRunResult(
        matched_edges=_NO_FRAME,
        raw_matched_edges=raw_matched_edges,
        pair_truth_eval=None,
        raw_pair_truth_eval=None,
        pair_truth_eval_detail=pair_truth_eval_detail,
        pruning_summary=_NO_FRAME,
        exact_match_summary=_NO_FRAME,
        clusters=_NO_FRAME,
        cluster_shape=_NO_FRAME,
        directional_coverage=_NO_FRAME,
        similarity_distribution=_NO_FRAME,
        candidate_pair_count=raw_matched_edges.height,
    )


def _truth_row(
    *,
    source_id: str,
    target_id: str,
    found: bool,
    similarity: float | None,
    rank: int | None,
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
        "name_equality": "never",
        "is_truth_pair": True,
        "found": found,
        "similarity": similarity,
        "rank": rank,
    }


# -- Direct tests of compute_pair_audit's own decision logic ---------------


def test_compute_pair_audit_refuses_a_run_made_with_a_per_target_cap() -> None:
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=5,
        min_similarity=0.1,
        max_candidates_per_source=None,
        max_candidates_per_target=1,
    )
    entry = StrategyRunEntry(
        label="tfidf/x",
        config=_config(strategy),
        result=_result(pair_truth_eval_detail=_EMPTY_DETAIL),
    )

    with pytest.raises(ValueError, match="max_candidates_per_target"):
        audit.compute_pair_audit(entry)


def test_compute_pair_audit_trusts_an_exact_backend_s_own_raw_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truth pair present in `raw_matched_edges` under an exact backend is
    read straight from there -- `rescore_missed_pairs` is never called."""
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=5,
        min_similarity=0.2,
        max_candidates_per_source=None,
        similarity_backend="sklearn",
    )
    detail = pl.DataFrame(
        [
            _truth_row(
                source_id="s1", target_id="t1", found=True, similarity=0.9, rank=1
            )
        ],
        schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS,
    )
    raw = pl.DataFrame(
        {
            "source_id": ["s1"],
            "target_id": ["t1"],
            "similarity": [0.9],
            "rank": [1],
            "country": ["gb"],
        },
        schema=_MATCHED_EDGES_SCHEMA,
    )
    entry = StrategyRunEntry(
        label="tfidf/x",
        config=_config(strategy),
        result=_result(pair_truth_eval_detail=detail, raw_matched_edges=raw),
    )

    def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
        raise AssertionError("an exact backend's own raw score must not be rescored")

    monkeypatch.setattr(audit, "rescore_missed_pairs", _boom)

    result = audit.compute_pair_audit(entry)

    row = result.row(0, named=True)
    assert row["exact_similarity"] == 0.9
    assert row["exact_rank"] == 1
    assert row["rescored"] is False
    assert row["exact_kept"] is True
    assert row["verdict"] == audit.VERDICT_FOUND


def test_compute_pair_audit_marks_a_found_pair_the_exact_top_k_would_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rank check applies to a pair the run *found*, not only a
    missed one: an approximate backend's own recorded rank is not trusted,
    so a pair whose true rank falls outside the run's `top_k` is counted as
    dropped by the exact scan."""
    strategy = BlockingStrategyConfig(
        representation="sbert",
        top_k=1,
        min_similarity=0.5,
        max_candidates_per_source=None,
        similarity_backend="hnsw",
    )
    detail = pl.DataFrame(
        [
            _truth_row(
                source_id="s1", target_id="t1", found=True, similarity=0.9, rank=1
            )
        ],
        schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS,
    )
    raw = pl.DataFrame(
        {
            "source_id": ["s1"],
            "target_id": ["t1"],
            "similarity": [0.9],
            "rank": [1],
            "country": ["gb"],
        },
        schema=_MATCHED_EDGES_SCHEMA,
    )
    entry = StrategyRunEntry(
        label="sbert/x",
        config=_config(strategy),
        result=_result(pair_truth_eval_detail=detail, raw_matched_edges=raw),
    )

    def fake_rescore(
        config: BlockingRunConfig,
        *,
        country: str,
        source_rows: pl.DataFrame,
        backend: str,
        top_k: int,
    ) -> pl.DataFrame:
        assert backend == "dense_brute"
        assert country == "gb"
        assert top_k == 1
        assert source_rows.get_column("system_uri").to_list() == ["s1"]
        # t1 ranks second, beyond the run's cut, so the rescore keeps t2 alone.
        return pl.DataFrame(
            {
                "source_id": ["s1"],
                "target_id": ["t2"],
                "similarity": [0.95],
                "rank": [1],
                "country": ["gb"],
            }
        )

    monkeypatch.setattr(
        audit,
        "load_country_frame",
        lambda *_a, **_k: pl.DataFrame({"system_uri": ["s1"]}),
    )
    monkeypatch.setattr(audit, "rescore_missed_pairs", fake_rescore)

    result = audit.compute_pair_audit(entry)

    row = result.row(0, named=True)
    assert row["run_found"] is True
    assert row["exact_similarity"] is None
    assert row["exact_rank"] is None
    assert row["exact_kept"] is False
    assert row["distance_to_threshold"] is None
    assert row["verdict"] == audit.VERDICT_BACKEND_ONLY
    assert row["rescored"] is True


def test_compute_pair_audit_rescores_a_pair_missing_from_raw_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truth pair the run never predicted at all is not in `raw_matched_edges`
    even under an exact backend, so it is always rescored."""
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=5,
        min_similarity=0.5,
        max_candidates_per_source=None,
        similarity_backend="sklearn",
    )
    detail = pl.DataFrame(
        [
            _truth_row(
                source_id="s1", target_id="t1", found=False, similarity=None, rank=None
            )
        ],
        schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS,
    )
    entry = StrategyRunEntry(
        label="tfidf/x",
        config=_config(strategy),
        result=_result(
            pair_truth_eval_detail=detail, raw_matched_edges=_EMPTY_MATCHED_EDGES
        ),
    )

    monkeypatch.setattr(
        audit,
        "load_country_frame",
        lambda *_a, **_k: pl.DataFrame({"system_uri": ["s1"]}),
    )
    monkeypatch.setattr(
        audit,
        "rescore_missed_pairs",
        lambda *_a, **_k: pl.DataFrame(
            {
                "source_id": ["s1"],
                "target_id": ["t1"],
                "similarity": [0.1],
                "rank": [3],
                "country": ["gb"],
            }
        ),
    )

    result = audit.compute_pair_audit(entry)

    row = result.row(0, named=True)
    assert row["exact_similarity"] == pytest.approx(0.1)
    assert row["exact_kept"] is False  # below min_similarity=0.5
    assert row["verdict"] == audit.VERDICT_NOT_KEPT_BY_EXACT_SCAN
    assert row["rescored"] is True


def test_summarize_pair_audit_computes_exact_scan_recall() -> None:
    schema = BLOCKING_ARTIFACT_SCHEMAS["pair_audit"]
    frame = pl.DataFrame(
        [
            {
                "label": "tfidf/x",
                "source_system": "gleif",
                "target_system": "gb",
                "country": "gb",
                "source_id": "s1",
                "target_id": "t1",
                "name_equality": "never",
                "run_found": True,
                "exact_similarity": 0.9,
                "exact_rank": 1,
                "exact_kept": True,
                "distance_to_threshold": 0.4,
                "verdict": audit.VERDICT_FOUND,
                "rescored": False,
            },
            {
                "label": "tfidf/x",
                "source_system": "gleif",
                "target_system": "gb",
                "country": "gb",
                "source_id": "s2",
                "target_id": "t2",
                "name_equality": "never",
                "run_found": False,
                "exact_similarity": 0.6,
                "exact_rank": 2,
                "exact_kept": True,
                "distance_to_threshold": 0.1,
                "verdict": audit.VERDICT_LOST_TO_BACKEND,
                "rescored": True,
            },
            {
                "label": "tfidf/x",
                "source_system": "gleif",
                "target_system": "gb",
                "country": "gb",
                "source_id": "s3",
                "target_id": "t3",
                "name_equality": "never",
                "run_found": False,
                "exact_similarity": 0.1,
                "exact_rank": 9,
                "exact_kept": False,
                "distance_to_threshold": -0.4,
                "verdict": audit.VERDICT_NOT_KEPT_BY_EXACT_SCAN,
                "rescored": True,
            },
        ],
        schema=schema,
    )

    summary = audit.summarize_pair_audit(frame)

    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["truth_pairs"] == 3
    assert row["run_found"] == 1
    assert row["exact_kept"] == 2
    assert row["lost_to_backend"] == 1
    assert row["not_kept_by_exact_scan"] == 1
    assert row["backend_only"] == 0
    assert row["exact_scan_recall"] == pytest.approx(2 / 3)


# -- A fixture run through the real pipeline, checked against an
# -- independent, fully unrestricted rescore of its whole truth population --


def _build_fixture(layer_fixture_dir) -> None:
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            # Found: an exact scan (and the run itself) both keep it.
            {
                "system_uri": "gleif:1",
                "name": "delta enterprises holdings",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
            # Missed by the run's own comparison-cleaning (a much closer
            # decoy target prunes it below `candidate_similarity_ratio`),
            # even though it clears the run's own top_k/min_similarity --
            # an exact scan at those settings would have kept it.
            {
                "system_uri": "gleif:2",
                "name": "beta holdings uk",
                "jurisdiction_code": "gb",
                "match_uri": "gb:2",
            },
            # Missed outright: no candidate at all, and genuinely dissimilar
            # from every target -- an exact scan would not keep it either.
            {
                "system_uri": "gleif:3",
                "name": "zzzz totally unrelated widgets",
                "jurisdiction_code": "gb",
                "match_uri": "gb:3",
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
                "name": "delta enterprises holdings",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:2",
                "name": "beta holdings united kingdom",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:3",
                "name": "quite a different company plc",
                "jurisdiction_code": "gb",
            },
            # The decoy: near-identical to gleif:2's own text, so its
            # candidate_similarity_ratio pruning drops the real match above.
            {
                "system_uri": "gb:4",
                "name": "beta holdings uk",
                "jurisdiction_code": "gb",
            },
        ],
    )


def _fixture_config(workspace_roots: WorkspaceRoots) -> BlockingRunConfig:
    source = load_dataset_descriptor(roots=workspace_roots, system="gleif")
    target = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=5,
        min_similarity=0.7,
        max_candidates_per_source=None,
        similarity_backend="sklearn",
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
        exact_name_filter=False,
        candidate_similarity_ratio=0.95,
    )
    return BlockingRunConfig(
        roots=workspace_roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
    )


def test_compute_pair_audit_matches_an_unrestricted_rescore_of_the_whole_population(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _build_fixture(layer_fixture_dir)
    config = _fixture_config(workspace_roots)
    result = execute_blocking_run(config)
    entry = StrategyRunEntry(label="tfidf/fixture", config=config, result=result)

    verdicts = {
        row["source_id"]: row["verdict"]
        for row in audit.compute_pair_audit(entry).iter_rows(named=True)
    }
    assert verdicts == {
        "gleif:1": audit.VERDICT_FOUND,
        "gleif:2": audit.VERDICT_LOST_TO_BACKEND,
        "gleif:3": audit.VERDICT_NOT_KEPT_BY_EXACT_SCAN,
    }

    pair_audit = audit.compute_pair_audit(entry)

    # The independent reference: every truth pair's source row rescored with
    # no shortlist at all (top_k already covers the whole target, so this is
    # the fixture-scale stand-in for the one real brute-force exact scan the
    # item's `Done when` names) -- the audit's own trusted-raw shortcut for
    # `gleif:1` must agree with it, not just with itself.
    detail = result.pair_truth_eval_detail
    assert detail is not None
    truth_source_ids = sorted(
        detail.filter(pl.col("is_truth_pair"))
        .get_column("source_id")
        .unique()
        .to_list()
    )
    full_source_frame = pl.concat(
        [
            _load_source_row(config, source_id=source_id)
            for source_id in truth_source_ids
        ],
        how="vertical_relaxed",
    )
    brute_force = rescore_missed_pairs(
        config, country="gb", source_rows=full_source_frame, backend="sklearn"
    )

    for row in pair_audit.iter_rows(named=True):
        brute_row = brute_force.filter(
            (pl.col("source_id") == row["source_id"])
            & (pl.col("target_id") == row["target_id"])
        )
        assert brute_row.height == 1
        assert row["exact_similarity"] == pytest.approx(
            brute_row.get_column("similarity").item()
        )
        assert row["exact_rank"] == brute_row.get_column("rank").item()

    summary = audit.summarize_pair_audit(pair_audit)
    assert summary.get_column("truth_pairs").sum() == 3
    assert summary.get_column("exact_kept").sum() == 2
    assert summary.get_column("lost_to_backend").sum() == 1
    assert summary.get_column("not_kept_by_exact_scan").sum() == 1


def _load_source_row(config: BlockingRunConfig, *, source_id: str) -> pl.DataFrame:
    from blocking.loader import load_country_frame

    frame = load_country_frame(config.source, country="gb")
    return frame.filter(pl.col("system_uri") == source_id)


# -- produce_pair_audit: a second audit of one run reuses the first --------


def test_produce_pair_audit_reuses_a_second_call(
    monkeypatch: pytest.MonkeyPatch, workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _build_fixture(layer_fixture_dir)
    config = _fixture_config(workspace_roots)
    keys = resolve_blocking_run_keys(config)
    with produce_blocking_run(config, keys=keys, invocation=["run"]) as staged:
        write_blocking_report(staged, execute_blocking_run(config, expected_keys=keys))
    run_dir = resolve_run_location_for(config, keys=keys).directory
    location = read_run_location(workspace_roots, run_dir)

    first_location, first_reused = audit.produce_pair_audit(
        workspace_roots, location, invocation=["audit"]
    )
    assert first_reused is False
    assert first_location.path(audit.AuditArtefact.PAIR_AUDIT).is_file()
    first_record = read_record(
        workspace_roots, reference_at(workspace_roots, first_location.directory)
    )
    assert first_record is not None

    def _boom(*_args: object, **_kwargs: object) -> StrategyRunEntry:
        raise AssertionError("a second audit of one run must not recompute it")

    monkeypatch.setattr(audit, "load_strategy_run_entry", _boom)

    second_location, second_reused = audit.produce_pair_audit(
        workspace_roots, location, invocation=["audit"]
    )

    assert second_reused is True
    assert second_location.directory == first_location.directory
    second_record = read_record(
        workspace_roots, reference_at(workspace_roots, second_location.directory)
    )
    assert second_record is not None
    assert second_record.started_at == first_record.started_at
