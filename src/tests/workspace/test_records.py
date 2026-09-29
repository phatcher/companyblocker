"""Direct tests for `workspace.records`.

A location is complete exactly when it holds a record written by `produce`; a
failed production leaves nothing; and the catalog answers by any segment and by
what consumed a reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workspace.identity import ConsumedReference, reference_key
from workspace.kind_layout import Kind
from workspace.records import (
    RECORD_FILENAME,
    ProductionError,
    consumers_of,
    iter_records,
    open_catalog,
    produce,
    read_record,
    select_recorded,
)
from workspace.reference import Reference, Side, locate, reference
from workspace.roots import WorkspaceRoots


def _profile() -> Reference:
    return reference(Kind.PERTURBATION, Side.PROFILE, name="en-lite", version="v1")


def _dataset(seed: str = "42") -> Reference:
    return reference(
        Kind.PERTURBATION,
        Side.DATA,
        source="ie",
        profile="en-lite",
        version="v1",
        seed=seed,
    )


def _run(representation: str = "tfidf", key: str = "aaaaaaaaaaaa") -> Reference:
    return reference(
        Kind.BLOCKING,
        Side.DATA,
        source="ie",
        target="gb",
        derivation="perturbed",
        profile="en-lite",
        version="v1",
        seed="42",
        representation=representation,
        key=key,
    )


def _author_profile(roots: WorkspaceRoots) -> ConsumedReference:
    locate(roots, _profile()).mkdir(parents=True, exist_ok=True)
    return ConsumedReference(_profile())


def _materialize(
    roots: WorkspaceRoots, *, text: str = "rows", digest: str = "d1"
) -> None:
    with produce(
        roots,
        _dataset(),
        inputs={"profile": _author_profile(roots)},
        parameters={"seed": 42},
        invocation=["materialize_perturbations.py", "--profile", "en-lite"],
    ) as production:
        (production.directory / "rows.txt").write_text(text, encoding="utf-8")
        production.content_digest = digest


def _block(
    roots: WorkspaceRoots, representation: str = "tfidf", key: str = "aaaaaaaaaaaa"
) -> None:
    with produce(
        roots,
        _run(representation, key),
        inputs={"source": ConsumedReference(_dataset(), content_digest="d1")},
        parameters={"k": 10, "representation": representation},
        invocation=["run_blocking.py"],
    ) as production:
        (production.directory / "summary.json").write_text("{}", encoding="utf-8")


def _hidden_siblings(location: Path) -> list[Path]:
    return [p for p in location.parent.iterdir() if p.name.startswith(".")]


def test_a_production_puts_its_files_and_its_record_in_place(
    workspace_roots: WorkspaceRoots,
):
    _materialize(workspace_roots)
    location = locate(workspace_roots, _dataset())

    record = read_record(workspace_roots, _dataset())

    assert (location / "rows.txt").read_text(encoding="utf-8") == "rows"
    assert record is not None
    assert record.uri == "perturbed://ie/en-lite/v1/42"
    assert record.inputs["profile"].uri == "perturbation://en-lite/v1"
    assert record.content_digest == "d1"
    assert record.invocation == ("materialize_perturbations.py", "--profile", "en-lite")
    assert record.key == reference_key(
        {"profile": ConsumedReference(_profile())}, {"seed": 42}
    )
    assert record.roots == workspace_roots.to_manifest()
    assert _hidden_siblings(location) == []


def test_a_record_round_trips_through_its_file(workspace_roots: WorkspaceRoots):
    _materialize(workspace_roots)
    path = locate(workspace_roots, _dataset()) / RECORD_FILENAME

    record = read_record(workspace_roots, _dataset())

    assert record is not None
    assert json.loads(record.to_json()) == json.loads(path.read_text(encoding="utf-8"))


def test_a_failed_production_leaves_no_location_and_no_record(
    workspace_roots: WorkspaceRoots,
):
    profile = _author_profile(workspace_roots)
    location = locate(workspace_roots, _dataset())

    with (
        pytest.raises(RuntimeError, match="boom"),
        produce(
            workspace_roots,
            _dataset(),
            inputs={"profile": profile},
            parameters={},
            invocation=["x"],
        ) as production,
    ):
        (production.directory / "rows.txt").write_text("half", encoding="utf-8")
        raise RuntimeError("boom")

    assert not location.exists()
    assert _hidden_siblings(location) == []
    assert read_record(workspace_roots, _dataset()) is None


def test_a_mutable_output_must_declare_its_content_digest(
    workspace_roots: WorkspaceRoots,
):
    profile = _author_profile(workspace_roots)

    with (
        pytest.raises(ProductionError, match="declare its content digest"),
        produce(
            workspace_roots,
            _dataset(),
            inputs={"profile": profile},
            parameters={},
            invocation=[],
        ),
    ):
        pass

    assert not locate(workspace_roots, _dataset()).exists()


def test_an_immutable_output_declares_no_content_digest(
    workspace_roots: WorkspaceRoots,
):
    _materialize(workspace_roots)

    with (
        pytest.raises(ProductionError, match="declares no digest"),
        produce(
            workspace_roots,
            _run(),
            inputs={"source": ConsumedReference(_dataset(), content_digest="d1")},
            parameters={},
            invocation=[],
        ) as production,
    ):
        production.content_digest = "x"


def test_a_production_is_refused_before_work_for_a_selection_or_a_missing_input(
    workspace_roots: WorkspaceRoots,
):
    with (
        pytest.raises(ProductionError, match="selection"),
        produce(
            workspace_roots,
            reference(Kind.BLOCKING, Side.DATA, target="gb"),
            inputs={},
            parameters={},
            invocation=[],
        ),
    ):
        pytest.fail("the body must not run")

    with (
        pytest.raises(ProductionError, match="do not exist"),
        produce(
            workspace_roots,
            _run(),
            inputs={"source": ConsumedReference(_dataset(), content_digest="d1")},
            parameters={},
            invocation=[],
        ),
    ):
        pytest.fail("the body must not run")


def test_an_immutable_output_already_complete_is_left_as_it_is(
    workspace_roots: WorkspaceRoots,
):
    _materialize(workspace_roots)
    _block(workspace_roots)
    first = read_record(workspace_roots, _run())

    with produce(
        workspace_roots,
        _run(),
        inputs={"source": ConsumedReference(_dataset(), content_digest="d1")},
        parameters={"k": 10, "representation": "tfidf"},
        invocation=["run_blocking.py"],
    ) as production:
        (production.directory / "summary.json").write_text(
            '{"second": true}', encoding="utf-8"
        )

    location = locate(workspace_roots, _run())
    assert (location / "summary.json").read_text(encoding="utf-8") == "{}"
    assert read_record(workspace_roots, _run()) == first
    assert _hidden_siblings(location) == []


def test_a_mutable_output_is_replaced_by_its_next_production(
    workspace_roots: WorkspaceRoots,
):
    _materialize(workspace_roots, text="first", digest="d1")
    _materialize(workspace_roots, text="second", digest="d2")
    location = locate(workspace_roots, _dataset())

    record = read_record(workspace_roots, _dataset())

    assert (location / "rows.txt").read_text(encoding="utf-8") == "second"
    assert record is not None and record.content_digest == "d2"
    assert _hidden_siblings(location) == []


def test_a_location_without_a_record_is_incomplete_and_is_replaced(
    workspace_roots: WorkspaceRoots,
):
    _materialize(workspace_roots)
    leftover = locate(workspace_roots, _run())
    leftover.mkdir(parents=True)
    (leftover / "summary.json").write_text("interrupted", encoding="utf-8")

    assert read_record(workspace_roots, _run()) is None
    _block(workspace_roots)
    assert (leftover / "summary.json").read_text(encoding="utf-8") == "{}"
    assert read_record(workspace_roots, _run()) is not None


def test_a_record_its_uri_does_not_locate_is_never_read(
    workspace_roots: WorkspaceRoots,
):
    _materialize(workspace_roots)
    stray = locate(workspace_roots, _dataset()).parent / ".42.abandoned"
    stray.mkdir()
    (stray / RECORD_FILENAME).write_text(
        (locate(workspace_roots, _dataset()) / RECORD_FILENAME).read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    assert [record.uri for record in iter_records(workspace_roots)] == [_dataset().uri]


def test_the_catalog_selects_by_any_segment_and_follows_what_consumed_a_reference(
    workspace_roots: WorkspaceRoots,
):
    _materialize(workspace_roots)
    _block(workspace_roots, "tfidf", "aaaaaaaaaaaa")
    _block(workspace_roots, "sbert", "bbbbbbbbbbbb")

    catalog = open_catalog(workspace_roots)

    assert select_recorded(catalog, representation="sbert") == [
        _run("sbert", "bbbbbbbbbbbb").uri
    ]
    assert select_recorded(catalog, source="ie") == [
        _run("sbert", "bbbbbbbbbbbb").uri,
        _run("tfidf", "aaaaaaaaaaaa").uri,
        _dataset().uri,
    ]
    assert select_recorded(catalog, target="gb", derivation="perturbed") == [
        _run("sbert", "bbbbbbbbbbbb").uri,
        _run("tfidf", "aaaaaaaaaaaa").uri,
    ]
    assert consumers_of(catalog, _dataset()) == [
        _run("sbert", "bbbbbbbbbbbb").uri,
        _run("tfidf", "aaaaaaaaaaaa").uri,
    ]
    assert consumers_of(catalog, _profile()) == [_dataset().uri]


def test_an_empty_workspace_has_an_empty_catalog(workspace_roots: WorkspaceRoots):
    assert select_recorded(open_catalog(workspace_roots)) == []


def test_a_producer_s_own_key_is_the_record_s_key(workspace_roots: WorkspaceRoots):
    """A producer that already derives its output's identity keeps it, so the
    record and the location it names carry one key."""
    _materialize(workspace_roots)

    with produce(
        workspace_roots,
        _run(),
        inputs={"source": ConsumedReference(_dataset(), content_digest="d1")},
        parameters={"k": 10},
        invocation=["run_blocking.py"],
        key="aaaaaaaaaaaa",
    ) as production:
        (production.directory / "summary.json").write_text("{}", encoding="utf-8")

    record = read_record(workspace_roots, _run())
    assert record is not None and record.key == "aaaaaaaaaaaa"
