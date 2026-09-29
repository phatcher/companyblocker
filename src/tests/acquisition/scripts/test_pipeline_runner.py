from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.pipeline_runner import (
    execute_canonical,
    execute_cleanse,
    prepare_tokenized_output_dir,
    resolve_promoted_tokenizer_path,
)
from tests.promoted_tokenizers import promoted_tokenizer_files
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    system_layer_dir,
)
from workspace.pointer import PointerError
from workspace.roots import WorkspaceRoots


def test_prepare_tokenized_output_dir_noop_when_no_legacy_sibling(
    tmp_path: Path, layer_fixture_dir
):
    tokenized_dir = layer_fixture_dir("ie", layer="tokenized")

    prepare_tokenized_output_dir(
        system_code="ie", tokenized_dir=tokenized_dir, input_file=None
    )  # should not raise

    assert tokenized_dir.exists()


def test_prepare_tokenized_output_dir_clears_stale_chunks_and_output(
    tmp_path: Path, layer_fixture_dir
):
    tokenized_dir = layer_fixture_dir("ie", layer="tokenized")
    chunks_dir = tokenized_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "ie-001.parquet").write_bytes(b"x")

    partition_dir = tokenized_dir / "jurisdiction_code=ie"
    partition_dir.mkdir(parents=True, exist_ok=True)
    (partition_dir / "part-00001.parquet").write_bytes(b"x")

    prepare_tokenized_output_dir(
        system_code="ie", tokenized_dir=tokenized_dir, input_file=None
    )

    assert not chunks_dir.exists()
    assert not partition_dir.exists()


def test_prepare_tokenized_output_dir_clears_family_split_primary_output(
    tmp_path: Path,
    layer_fixture_dir,
):
    """company_tokenize mirrors its cleansed/ input's own relative path, so
    real tokenized output lands under a `primary/` family directory, not
    directly at tokenized_dir's own top level -- stale output there must be
    cleared too, not only the pre-split top-level layout the other test
    above exercises.
    """
    tokenized_dir = layer_fixture_dir("ie", layer="tokenized")
    partition_dir = tokenized_dir / "primary" / "jurisdiction_code=ie"
    partition_dir.mkdir(parents=True, exist_ok=True)
    (partition_dir / "part-00001.parquet").write_bytes(b"x")

    prepare_tokenized_output_dir(
        system_code="ie", tokenized_dir=tokenized_dir, input_file=None
    )

    assert not (tokenized_dir / "primary").exists()


def _make_stage_fn(
    *, paths=None, sleep_seconds: float = 0.0, exc: Exception | None = None
):
    """Build a fake canonical-stage callable matching the `*_fn` signatures
    `execute_canonical` calls with (positional system_code, keyword-only
    everything else)."""

    def _fn(system_code: str, **_kwargs) -> list[Path]:
        if sleep_seconds:
            time.sleep(sleep_seconds)
        if exc is not None:
            raise exc
        return list(paths or [])

    return _fn


def _execute_canonical_kwargs(roots: WorkspaceRoots, **overrides) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "system_code": "gleif",
        "roots": roots,
        "run_date": None,
        "chunk_size": 100_000,
        "allow_research": False,
        "derive_system_name_rows_fn": _make_stage_fn(
            paths=[Path("names-shard-1.parquet")]
        ),
        "canonicalize_system_shards_fn": _make_stage_fn(
            paths=[Path("shard-1.parquet"), Path("shard-2.parquet")]
        ),
        "canonicalize_system_name_rows_fn": _make_stage_fn(
            paths=[Path("names-1.parquet")]
        ),
        "canonicalize_system_primary_name_override_fn": _make_stage_fn(
            paths=[Path("override-1.parquet")]
        ),
        "canonicalize_system_successor_chain_fn": _make_stage_fn(
            paths=[Path("successor-1.parquet")]
        ),
        "finalize_canonical_name_rows_fn": _make_stage_fn(
            paths=[Path("names-final-1.parquet")]
        ),
    }
    kwargs.update(overrides)
    return kwargs


def test_execute_canonical_success_runs_all_stages_in_correct_order(
    workspace_roots: WorkspaceRoots,
):
    calls: list[str] = []

    def _tracking_fn(label: str, *, paths=None):
        def _fn(system_code: str, **_kwargs):
            calls.append(label)
            return list(paths or [])

        return _fn

    kwargs = _execute_canonical_kwargs(
        workspace_roots,
        derive_system_name_rows_fn=_tracking_fn("derive"),
        canonicalize_system_shards_fn=_tracking_fn(
            "entity", paths=[Path("a.parquet"), Path("b.parquet")]
        ),
        canonicalize_system_name_rows_fn=_tracking_fn("name_rows"),
        canonicalize_system_primary_name_override_fn=_tracking_fn("override"),
        canonicalize_system_successor_chain_fn=_tracking_fn("successor_chain"),
        finalize_canonical_name_rows_fn=_tracking_fn("finalize_names"),
    )

    failed, count = execute_canonical(**kwargs)

    assert failed is False
    # _run_stage_and_report always returns a literal 0 count on success,
    # pre-existing and unrelated to this change (its only caller already
    # discards this tuple element); assert the returned shape stays intact
    # rather than asserting a written-path count it never carried.
    assert count == 0
    assert set(calls) == {
        "derive",
        "entity",
        "name_rows",
        "override",
        "successor_chain",
        "finalize_names",
    }
    # The downstream passes must only run after both concurrent passes
    # have completed.
    assert calls.index("override") > calls.index("entity")
    assert calls.index("override") > calls.index("name_rows")
    # And name finalization must follow the successor chain, which
    # appends rows to the very sidecar it consolidates.
    assert calls.index("finalize_names") > calls.index("successor_chain")


def test_execute_canonical_name_row_pass_failure_reported_independently(
    capsys, workspace_roots: WorkspaceRoots
):
    kwargs = _execute_canonical_kwargs(
        workspace_roots,
        canonicalize_system_shards_fn=_make_stage_fn(
            paths=[Path("a.parquet"), Path("b.parquet"), Path("c.parquet")],
            sleep_seconds=0.1,
        ),
        canonicalize_system_name_rows_fn=_make_stage_fn(
            exc=RuntimeError("bad name row")
        ),
        canonicalize_system_primary_name_override_fn=_make_stage_fn(),
        canonicalize_system_successor_chain_fn=_make_stage_fn(),
    )
    override_calls: list[str] = []
    successor_calls: list[str] = []

    def _tracking_override(system_code: str, **_kwargs):
        override_calls.append(system_code)
        return []

    def _tracking_successor(system_code: str, **_kwargs):
        successor_calls.append(system_code)
        return []

    kwargs["canonicalize_system_primary_name_override_fn"] = _tracking_override
    kwargs["canonicalize_system_successor_chain_fn"] = _tracking_successor

    failed, count = execute_canonical(**kwargs)

    assert failed is True
    # count is always 0 -- _run_stage_and_report never returns a real
    # written-path count on either branch; only failed/success matters here.
    assert count == 0
    assert override_calls == []
    assert successor_calls == []

    captured = capsys.readouterr()
    assert "canonical name rows" in captured.out
    assert "bad name row" in captured.out


def test_execute_canonical_entity_pass_failure_reported_independently(
    capsys, workspace_roots: WorkspaceRoots
):
    kwargs = _execute_canonical_kwargs(
        workspace_roots,
        canonicalize_system_shards_fn=_make_stage_fn(
            exc=FileNotFoundError("no shard files")
        ),
        canonicalize_system_name_rows_fn=_make_stage_fn(
            paths=[Path("names-1.parquet")], sleep_seconds=0.1
        ),
    )
    override_calls: list[str] = []
    successor_calls: list[str] = []

    def _tracking_override(system_code: str, **_kwargs):
        override_calls.append(system_code)
        return []

    def _tracking_successor(system_code: str, **_kwargs):
        successor_calls.append(system_code)
        return []

    kwargs["canonicalize_system_primary_name_override_fn"] = _tracking_override
    kwargs["canonicalize_system_successor_chain_fn"] = _tracking_successor

    failed, count = execute_canonical(**kwargs)

    assert failed is True
    # The entity pass never wrote anything, regardless of the name-row
    # chain's own success.
    assert count == 0
    assert override_calls == []
    assert successor_calls == []

    captured = capsys.readouterr()
    assert "Stage canonical failed" in captured.out
    assert "no shard files" in captured.out


def test_execute_canonical_derive_failure_short_circuits_name_row_pass(
    capsys, workspace_roots: WorkspaceRoots
):
    name_rows_calls: list[str] = []

    def _tracking_name_rows(system_code: str, **_kwargs):
        name_rows_calls.append(system_code)
        return []

    kwargs = _execute_canonical_kwargs(
        workspace_roots,
        derive_system_name_rows_fn=_make_stage_fn(exc=RuntimeError("bad derive")),
        canonicalize_system_name_rows_fn=_tracking_name_rows,
        canonicalize_system_shards_fn=_make_stage_fn(
            paths=[Path("a.parquet")], sleep_seconds=0.1
        ),
    )

    failed, count = execute_canonical(**kwargs)

    assert failed is True
    assert count == 0
    assert name_rows_calls == []  # never reached: derive failed first

    captured = capsys.readouterr()
    assert "shard name-row derivation" in captured.out
    assert "bad derive" in captured.out


def test_execute_canonical_runs_entity_and_name_row_passes_concurrently(
    workspace_roots: WorkspaceRoots,
):
    sleep_seconds = 0.2
    kwargs = _execute_canonical_kwargs(
        workspace_roots,
        canonicalize_system_shards_fn=_make_stage_fn(
            paths=[Path("a.parquet")], sleep_seconds=sleep_seconds
        ),
        canonicalize_system_name_rows_fn=_make_stage_fn(
            paths=[Path("n.parquet")], sleep_seconds=sleep_seconds
        ),
    )

    start = time.monotonic()
    failed, _ = execute_canonical(**kwargs)
    elapsed = time.monotonic() - start

    assert failed is False
    # If the two passes ran sequentially this would take >= 2 * sleep_seconds
    # (plus the derive/override/successor-chain stages on top); concurrent
    # execution keeps it well under that.
    assert elapsed < 1.5 * sleep_seconds


def test_execute_cleanse_never_passes_a_narrowed_company_type_rule_set(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
):
    """execute_cleanse must not resolve company-type rules from the system
    code and hand them to `cleanse_canonical_view` -- that narrows cleansing
    to one jurisdiction's suffixes, or to nothing at all for a system code
    naming no jurisdiction. Leaving `company_type_regex`/
    `company_type_mapping` out of the call lets `CleanseConfig`'s own
    default resolve the full, multi-jurisdiction rule set instead, so a
    later caller can't silently narrow it again.
    """
    captured_kwargs: dict[str, object] = {}

    def fake_cleanse_canonical_view(**kwargs):
        captured_kwargs.update(kwargs)
        return 3

    system_plan = SimpleNamespace(
        code="gb",
        company_type_column="CompanyCategory",
        personal_owner_markers=None,
        resources=[],
    )

    failed, cleansed_rows = execute_cleanse(
        system_code="gb",
        roots=workspace_roots,
        run_date=None,
        input_file=None,
        company_col="CompanyName",
        char_whitelist=r"[^a-z0-9\s!&]",
        resolve_effective_and_tokens=lambda *, system_plan: None,
        configure_fn=lambda *, country, roots: SimpleNamespace(
            cleansed_dir=system_layer_dir(roots, country, layer=CLEANSED_LAYER_NAME)
        ),
        resolve_input_dir_fn=lambda *, roots, run_date, system: system_layer_dir(
            roots, system, layer=CANONICAL_LAYER_NAME
        ),
        get_system_plan_fn=lambda system_code: system_plan,
        get_system_company_type_mapping_fn=lambda system_code: {},
        cleanse_canonical_view_fn=fake_cleanse_canonical_view,
    )

    assert failed is False
    assert cleansed_rows == 3
    assert "company_type_regex" not in captured_kwargs
    assert "company_type_mapping" not in captured_kwargs


def test_resolve_promoted_tokenizer_path_names_the_promoted_model(
    workspace_roots: WorkspaceRoots,
):
    country = promoted_tokenizer_files(workspace_roots, system="fr").model
    shared = promoted_tokenizer_files(
        workspace_roots, system=None, trainer="sentencepiece"
    ).model

    assert (
        resolve_promoted_tokenizer_path(
            roots=workspace_roots, scope="country", system="fr"
        )
        == country
    )
    assert (
        resolve_promoted_tokenizer_path(
            roots=workspace_roots, scope="global", trainer="sentencepiece"
        )
        == shared
    )


def test_resolve_promoted_tokenizer_path_is_none_only_when_nothing_is_promoted(
    workspace_roots: WorkspaceRoots,
):
    """Nothing promoted leaves Tokenize on the packaged default; a promoted
    candidate missing from the store must not fall back to it."""
    assert (
        resolve_promoted_tokenizer_path(
            roots=workspace_roots, scope="country", system="fr"
        )
        is None
    )

    files = promoted_tokenizer_files(workspace_roots, system="fr")
    shutil.rmtree(files.directory)

    with pytest.raises(PointerError, match="absent"):
        resolve_promoted_tokenizer_path(
            roots=workspace_roots, scope="country", system="fr"
        )
