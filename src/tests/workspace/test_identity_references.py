"""Direct tests for `workspace.identity`'s keys built from references.

A run's key comes from the parsed fields of the references it consumed and its
parameters: never the roots, never the rendered URI, and for a mutable
reference always its content.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from workspace import reference as reference_module
from workspace.identity import (
    ConsumedReference,
    ContentDigestError,
    digest_directory,
    reference_key,
)
from workspace.kind_layout import Kind
from workspace.reference import (
    SLUG,
    Layout,
    Part,
    Reference,
    Segment,
    Side,
    locate,
    reference,
    reference_at,
)
from workspace.roots import default_workspace_roots


def _run(**overrides: str) -> Reference:
    values = {
        "source": "gleif",
        "target": "gb",
        "representation": "tfidf",
        "key": "0123456789ab",
        **overrides,
    }
    return reference(Kind.BLOCKING, Side.DATA, **values)


def _dataset(seed: str = "42") -> Reference:
    return reference(
        Kind.PERTURBATION,
        Side.DATA,
        source="ie",
        profile="en-lite",
        version="v1",
        seed=seed,
    )


def _key(run: Reference, parameters: dict[str, object] | None = None) -> str:
    return reference_key(
        {"pairs": ConsumedReference(run)},
        parameters if parameters is not None else {"k": 10},
    )


def test_a_key_does_not_depend_on_where_the_roots_are(tmp_path: Path):
    near = default_workspace_roots(tmp_path / "near")
    far = default_workspace_roots(tmp_path / "far" / "away")
    run = _run()

    read_near = reference_at(near, locate(near, run))
    read_far = reference_at(far, locate(far, run))

    assert _key(read_near) == _key(read_far)


@pytest.mark.parametrize(
    "changed",
    [
        {"target": "ie"},
        {"dataset": "names"},
        {"source": "wikidata"},
        {"representation": "sbert"},
        {"key": "ba9876543210"},
    ],
)
def test_a_key_moves_with_each_identity_segment(changed: dict[str, str]):
    assert _key(_run(**changed)) != _key(_run())


def test_a_key_moves_with_a_role_and_a_parameter():
    run = ConsumedReference(_run())

    base = reference_key({"pairs": run}, {"k": 10})

    assert reference_key({"truth": run}, {"k": 10}) != base
    assert reference_key({"pairs": run}, {"k": 11}) != base


def test_a_parameter_at_its_default_moves_no_key():
    run = ConsumedReference(_run())

    assert reference_key(
        {"pairs": run}, {"k": 10, "rerank": False}, defaults={"rerank": False}
    ) == reference_key({"pairs": run}, {"k": 10})


def test_a_segment_outside_identity_moves_no_key(monkeypatch):
    layout = Layout(
        scheme="probe",
        kind=Kind.BLOCKING,
        side=Side.PROFILE,
        segments=(Segment("name", SLUG), Segment("label", SLUG, identity=False)),
        path=(Part(("name",)), Part(("label",))),
        immutable=True,
    )
    monkeypatch.setitem(
        reference_module._LAYOUTS, (Kind.BLOCKING, Side.PROFILE), [layout]
    )

    def keyed(label: str) -> str:
        ref = reference(Kind.BLOCKING, Side.PROFILE, name="gb", label=label)
        return reference_key({"input": ConsumedReference(ref)}, {})

    assert keyed("first") == keyed("second")


def test_a_key_hashes_parsed_fields_not_the_rendered_uri(monkeypatch):
    before = _key(_run())
    monkeypatch.setattr(
        reference_module, "render_reference", lambda ref: "renamed://" + ref.side.value
    )

    assert _key(_run()) == before


def test_a_mutable_reference_needs_its_content_and_moves_with_it():
    dataset = _dataset()

    with pytest.raises(ContentDigestError, match="mutable"):
        reference_key({"source": ConsumedReference(dataset)}, {})
    assert reference_key(
        {"source": ConsumedReference(dataset, content_digest="aa")}, {}
    ) != reference_key({"source": ConsumedReference(dataset, content_digest="bb")}, {})


def test_an_immutable_reference_refuses_a_content_digest():
    with pytest.raises(ContentDigestError, match="immutable"):
        reference_key({"pairs": ConsumedReference(_run(), content_digest="aa")}, {})


def test_a_draft_is_mutable_where_its_promoted_version_is_not():
    promoted = reference(Kind.PERTURBATION, Side.PROFILE, name="en-lite", version="v2")
    draft = reference(
        Kind.PERTURBATION, Side.PROFILE, name="en-lite", version="v2", stage="draft"
    )

    reference_key({"profile": ConsumedReference(promoted)}, {})
    with pytest.raises(ContentDigestError, match="mutable"):
        reference_key({"profile": ConsumedReference(draft)}, {})


def test_a_selection_is_not_a_key_input():
    selection = reference(Kind.BLOCKING, Side.DATA, target="gb")

    with pytest.raises(ValueError, match="selection"):
        reference_key({"pairs": ConsumedReference(selection)}, {})


def test_digest_directory_reads_paths_and_bytes_not_timestamps(tmp_path: Path):
    directory = tmp_path / "draft"
    (directory / "nested").mkdir(parents=True)
    (directory / "profile.json").write_text("{}", encoding="utf-8")
    (directory / "nested" / "notes.txt").write_text("a", encoding="utf-8")
    before = digest_directory(directory)

    os.utime(directory / "profile.json", (0, 0))
    assert digest_directory(directory) == before

    (directory / "nested" / "notes.txt").write_text("b", encoding="utf-8")
    edited = digest_directory(directory)
    assert edited != before

    (directory / "nested" / "notes.txt").rename(directory / "nested" / "other.txt")
    assert digest_directory(directory) != edited
