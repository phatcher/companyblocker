from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from acquisition.sharding_wikidata import (
    WikidataShardHandler,
    _wikidata_projected_spec_sidecar_path,
)
from workspace.roots import WorkspaceRoots

_HANDLER = WikidataShardHandler()


def test_resolve_wikidata_shard_source_prefers_projected_company_dump(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    source_dir.mkdir(parents=True, exist_ok=True)
    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "wikidata-all-2026-06-27.json.bz2"
    source_path.write_bytes(b"raw")

    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text('{"id":"Q1"}\n', encoding="utf-8")

    resolved_path, source_format = _HANDLER.resolve_shard_source(
        resource_name="wikidata_entities_latest_all",
        snapshot_date="2026-06-27",
        prepare_root=prepare_root,
        source_path=source_path,
        roots=workspace_roots,
    )

    assert resolved_path == projected_path
    assert source_format == "jsonl"


def test_resolve_wikidata_shard_source_prefers_projected_company_dump_for_gz_source(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    source_dir.mkdir(parents=True, exist_ok=True)
    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "wikidata-all.json.gz"
    source_path.write_bytes(b"raw")

    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text('{"id":"Q1"}\n', encoding="utf-8")

    resolved_path, source_format = _HANDLER.resolve_shard_source(
        resource_name="wikidata_entities_latest_all",
        snapshot_date="2026-06-27",
        prepare_root=prepare_root,
        source_path=source_path,
        roots=workspace_roots,
    )

    assert resolved_path == projected_path
    assert source_format == "jsonl"


def test_resolve_wikidata_shard_source_requires_projected_company_dump(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    source_dir.mkdir(parents=True, exist_ok=True)
    prepare_root = layer_fixture_dir("wikidata", layer="prepare")

    source_path = source_dir / "wikidata-all-2026-06-27.json.bz2"
    source_path.write_bytes(b"raw")

    try:
        _HANDLER.resolve_shard_source(
            resource_name="wikidata_entities_latest_all",
            snapshot_date="2026-06-27",
            prepare_root=prepare_root,
            source_path=source_path,
            roots=workspace_roots,
        )
    except FileNotFoundError as exc:
        assert "Wikidata prepare output not found" in str(exc)
    else:
        raise AssertionError(
            "Expected FileNotFoundError when Wikidata prepare output is missing"
        )


def test_resolve_wikidata_shard_source_accepts_partitioned_chunk_layout(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    source_dir.mkdir(parents=True, exist_ok=True)
    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    chunk_dir = prepare_dir / "wikidata-companies.chunks"
    partition_dir = chunk_dir / "000000-000999"
    partition_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "wikidata-all-2026-06-27.json.bz2"
    source_path.write_bytes(b"raw")
    (partition_dir / "wikidata-companies-part-000001.jsonl").write_text(
        '{"id":"Q1"}\n',
        encoding="utf-8",
    )

    resolved_path, source_format = _HANDLER.resolve_shard_source(
        resource_name="wikidata_entities_latest_all",
        snapshot_date="2026-06-27",
        prepare_root=prepare_root,
        source_path=source_path,
        roots=workspace_roots,
    )

    assert resolved_path == chunk_dir
    assert source_format == "jsonl"


def test_prepare_shard_input_stream_mode_resumes_when_chunks_exist(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    chunk_dir = prepare_dir / "wikidata-companies.chunks"
    partition_dir = chunk_dir / "000000-000999"
    partition_dir.mkdir(parents=True, exist_ok=True)
    (partition_dir / "wikidata-companies-part-000001.jsonl").write_text(
        '{"id":"Q1"}\n',
        encoding="utf-8",
    )

    captured: dict[str, object] = {"called": False}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="stream",
        stream_resume=True,
        projection_defaults=None,
    ) -> int:
        captured["called"] = True
        captured["source_path"] = source_path
        captured["destination_path"] = destination_path
        captured["phase1_wiring_mode"] = phase1_wiring_mode
        captured["stream_resume"] = stream_resume
        return 1

    resource = SimpleNamespace(name="wikidata_entities_latest_all")

    prepared = _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=lambda _message: None,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    # `prepare_shard_input` returns None when it declines the resource; this
    # test's resource is one it handles.
    assert prepared is not None
    snapshot_dir, resolved_source = prepared

    assert snapshot_dir == acquire_dir
    assert captured["called"] is True
    assert captured["source_path"] == acquire_path
    assert captured["destination_path"] == (prepare_dir / "wikidata-companies.jsonl")
    assert captured["phase1_wiring_mode"] == "stream"
    assert captured["stream_resume"] is True
    assert resolved_source == chunk_dir


def test_prepare_shard_input_stream_mode_reuses_existing_legacy_projected_file(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """A non-empty flat projected file is a durable, reusable artifact --
    rebuilding it from scratch whenever a (Rust-engine-only, effectively
    unreachable) chunk directory happens to be absent would throw away work
    that can otherwise cost hours to regenerate. See downloader_wikidata.py's
    _resolve_wikidata_projection_resume_defaults for why chunked output never
    actually materializes through this integration.
    """
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text('{"id":"Q-existing"}\n', encoding="utf-8")

    captured: dict[str, object] = {"called": False}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="stream",
        stream_resume=True,
        projection_defaults=None,
    ) -> int:
        captured["called"] = True
        return 1

    resource = SimpleNamespace(name="wikidata_entities_latest_all")

    prepared = _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=lambda _message: None,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    # `prepare_shard_input` returns None when it declines the resource; this
    # test's resource is one it handles.
    assert prepared is not None
    snapshot_dir, resolved_source = prepared

    assert snapshot_dir == acquire_dir
    assert captured["called"] is False
    assert resolved_source == projected_path
    assert projected_path.read_text(encoding="utf-8") == '{"id":"Q-existing"}\n'


def test_prepare_shard_input_stream_mode_ignores_empty_legacy_projected_file(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """A zero-byte flat file (for example: left behind by a run that was
    killed before writing anything) must not be treated as a valid, reusable
    artifact -- it must trigger a real extraction, not be silently accepted.
    """
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text("", encoding="utf-8")

    captured: dict[str, object] = {"called": False}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="stream",
        stream_resume=True,
        projection_defaults=None,
    ) -> int:
        captured["called"] = True
        destination_path.write_text('{"id":"Q-fresh"}\n', encoding="utf-8")
        return 1

    resource = SimpleNamespace(name="wikidata_entities_latest_all")

    _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=lambda _message: None,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    assert captured["called"] is True


def test_prepare_shard_input_applies_projection_engine_override(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="stream",
        stream_resume=True,
        projection_defaults=None,
    ) -> int:
        captured["projection_defaults"] = projection_defaults
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text('{"id":"Q1"}\n', encoding="utf-8")
        return 1

    resource = SimpleNamespace(
        name="wikidata_entities_latest_all",
        projection_defaults={
            "engine": "python",
            "binary_path": "tools/bin/wikisieve.exe",
            "output_mode": "jsonl",
        },
    )

    _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="file",
        prepare_stream_resume=False,
        projection_defaults_override={"engine": "wikisieve"},
        emit=lambda _message: None,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    assert isinstance(captured.get("projection_defaults"), dict)
    projection_defaults = cast(dict[str, object], captured["projection_defaults"])
    assert projection_defaults["engine"] == "wikisieve"


def test_prepare_shard_input_leaves_the_live_projected_file_in_place_for_the_extractor(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """A max_companies override that the cached file doesn't yet satisfy
    extracts while a real projected file already exists. Nothing moves that
    file aside first: the extractor replaces it only with a finished,
    verified extract, so a run that fails leaves it where it was."""
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text('{"id":"Q-existing"}\n', encoding="utf-8")

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="stream",
        stream_resume=True,
        projection_defaults=None,
    ) -> int:
        assert destination_path.read_text(encoding="utf-8") == '{"id":"Q-existing"}\n'
        destination_path.write_text('{"id":"Q-fresh"}\n', encoding="utf-8")
        return 1

    resource = SimpleNamespace(name="wikidata_entities_latest_all")

    _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=1_000_000,  # far more than the single cached row
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=lambda _message: None,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    assert projected_path.read_text(encoding="utf-8") == '{"id":"Q-fresh"}\n'
    assert list(prepare_dir.glob("*.bak")) == []


def test_prepare_shard_input_force_reextracts_despite_existing_projected_file(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """`should_extract` must not treat an existing projected artifact as
    unconditionally authoritative when re-projection is explicitly
    requested -- previously the only way through was moving the artifact
    aside by hand.
    """
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text('{"id":"Q-stale"}\n', encoding="utf-8")

    captured: dict[str, object] = {"called": False}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="stream",
        stream_resume=True,
        projection_defaults=None,
    ) -> int:
        captured["called"] = True
        destination_path.write_text('{"id":"Q-fresh"}\n', encoding="utf-8")
        return 1

    resource = SimpleNamespace(name="wikidata_entities_latest_all")
    messages: list[str] = []

    prepared = _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=messages.append,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
        force=True,
    )

    assert prepared is not None
    assert captured["called"] is True
    assert projected_path.read_text(encoding="utf-8") == '{"id":"Q-fresh"}\n'
    assert list(prepare_dir.glob("*.bak")) == []
    assert any("force requested; re-extracting" in message for message in messages)
    assert not any(
        "reusing existing projected artifact" in message for message in messages
    )


def test_prepare_shard_input_records_spec_sidecar_and_reuses_silently_when_it_matches(
    tmp_path: Path, layer_fixture_dir, capsys, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir / "wikidata-companies.jsonl"

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="stream",
        stream_resume=True,
        projection_defaults=None,
    ) -> int:
        destination_path.write_text('{"id":"Q1"}\n', encoding="utf-8")
        return 1

    resource = SimpleNamespace(
        name="wikidata_entities_latest_all",
        projection_defaults={"engine": "python"},
    )

    # First run: nothing exists yet, so it extracts and must record a spec
    # sidecar describing what produced the artifact.
    _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=lambda _message: None,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    sidecar_path = _wikidata_projected_spec_sidecar_path(projected_path)
    assert sidecar_path.exists()
    assert "python" in sidecar_path.read_text(encoding="utf-8")

    # Second run: same configured spec, existing artifact -- must reuse
    # silently (info-level), not warn.
    messages: list[str] = []
    _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=messages.append,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    assert any("reusing existing projected artifact" in message for message in messages)
    assert "[warn]" not in capsys.readouterr().out


def test_prepare_shard_input_reuse_warns_when_recorded_spec_no_longer_matches(
    tmp_path: Path, layer_fixture_dir, capsys, workspace_roots: WorkspaceRoots
):
    """A projected artifact was built under one spec, the configured spec
    has since changed, and reuse would otherwise be a silent no-op
    indistinguishable from an intended resume.
    """
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text('{"id":"Q-existing"}\n', encoding="utf-8")

    sidecar_path = _wikidata_projected_spec_sidecar_path(projected_path)
    sidecar_path.write_text('{"engine": "wikisieve"}', encoding="utf-8")

    def fake_extract(*args, **kwargs) -> int:
        raise AssertionError("should not extract when reusing")

    resource = SimpleNamespace(
        name="wikidata_entities_latest_all",
        projection_defaults={"engine": "python"},
    )

    messages: list[str] = []
    prepared = _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=messages.append,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    assert prepared is not None
    # The artifact is still reused (no force requested) -- only the message
    # channel changes, not the decision.
    assert projected_path.read_text(encoding="utf-8") == '{"id":"Q-existing"}\n'
    assert not any(
        "reusing existing projected artifact" in message for message in messages
    )
    printed = capsys.readouterr().out
    assert "[warn]" in printed
    assert "recorded projection spec no longer matches" in printed


def test_prepare_shard_input_reuse_stays_silent_when_no_spec_sidecar_recorded(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """A legacy artifact that predates spec recording has no sidecar to
    compare against -- unknown must not be treated as mismatched, or every
    pre-existing artifact would warn on its very next reuse.
    """
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-27"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    acquire_path = acquire_dir / "wikidata-all.json.gz"
    acquire_path.write_bytes(b"raw")

    prepare_root = layer_fixture_dir("wikidata", layer="prepare")
    prepare_dir = prepare_root / "2026-06-27"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir / "wikidata-companies.jsonl"
    projected_path.write_text('{"id":"Q-existing"}\n', encoding="utf-8")

    def fake_extract(*args, **kwargs) -> int:
        raise AssertionError("should not extract when reusing")

    resource = SimpleNamespace(name="wikidata_entities_latest_all")
    messages: list[str] = []

    _HANDLER.prepare_shard_input(
        resource=resource,
        acquire_root=layer_fixture_dir("wikidata", layer="acquire"),
        prepare_root=prepare_root,
        effective_run_date="2026-06-27",
        run_date_provided=True,
        roots=workspace_roots,
        max_companies=None,
        max_lines=None,
        prepare_wiring_mode="stream",
        prepare_stream_resume=True,
        projection_defaults_override=None,
        emit=messages.append,
        resolve_source_artifact=lambda **_kwargs: (acquire_dir, acquire_path),
        extract_two_pass=fake_extract,
    )

    assert any("reusing existing projected artifact" in message for message in messages)
