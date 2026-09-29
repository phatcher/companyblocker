"""Direct tests for `workspace.pointer`.

What is pinned is what a reader and a promotion rely on: a pointer resolves a
selection to the one reference promoted for it, refuses by name whatever it
cannot load, records what each promotion replaced, and is written so a diff of
the tracked file shows the decision and nothing else.
"""

from __future__ import annotations

import json

import pytest

from workspace.kind_layout import Kind
from workspace.pointer import (
    NothingPromotedError,
    PointerError,
    pointer_path,
    promote,
    promoted_reference,
)
from workspace.reference import Reference, Side, locate, reference
from workspace.roots import WorkspaceRoots

SLOT = reference(Kind.TOKENIZER, Side.DATA, scope="ie", tokenizer_id="wordpiece")


def _candidate(roots: WorkspaceRoots, key: str, *, scope: str = "ie") -> Reference:
    ref = reference(
        Kind.TOKENIZER, Side.DATA, scope=scope, tokenizer_id="wordpiece", key=key
    )
    locate(roots, ref).mkdir(parents=True)
    return ref


def test_the_pointer_is_one_tracked_file_per_kind_under_config(
    workspace_roots: WorkspaceRoots,
):
    assert pointer_path(workspace_roots, Kind.TOKENIZER) == (
        workspace_roots.config / "tokenizers.json"
    )


def test_a_promoted_reference_resolves_and_records_what_it_replaced(
    workspace_roots: WorkspaceRoots,
):
    first = _candidate(workspace_roots, "aaaa_v1")
    second = _candidate(workspace_roots, "bbbb_v2")

    assert promote(workspace_roots, SLOT, first) is None
    assert promoted_reference(workspace_roots, SLOT) == first
    assert promote(workspace_roots, SLOT, second) == first
    assert promoted_reference(workspace_roots, SLOT) == second

    written = pointer_path(workspace_roots, Kind.TOKENIZER).read_text(encoding="utf-8")
    assert json.loads(written) == {
        "promoted": {SLOT.uri: {"reference": second.uri, "replaced": first.uri}}
    }
    assert written.endswith("}\n") and "\r" not in written


def test_promoting_the_promoted_reference_again_changes_nothing(
    workspace_roots: WorkspaceRoots,
):
    first = _candidate(workspace_roots, "aaaa_v1")
    second = _candidate(workspace_roots, "bbbb_v2")
    promote(workspace_roots, SLOT, first)
    promote(workspace_roots, SLOT, second)
    path = pointer_path(workspace_roots, Kind.TOKENIZER)
    before = path.read_bytes()

    assert promote(workspace_roots, SLOT, second) == first
    assert path.read_bytes() == before


def test_each_selection_keeps_its_own_promotion(workspace_roots: WorkspaceRoots):
    other_slot = reference(
        Kind.TOKENIZER, Side.DATA, scope="gb", tokenizer_id="wordpiece"
    )
    ie = _candidate(workspace_roots, "aaaa_v1")
    gb = _candidate(workspace_roots, "aaaa_v1", scope="gb")

    promote(workspace_roots, SLOT, ie)
    promote(workspace_roots, other_slot, gb)

    assert promoted_reference(workspace_roots, SLOT) == ie
    assert promoted_reference(workspace_roots, other_slot) == gb


def test_each_profile_keeps_its_own_reference_and_what_it_replaced(
    workspace_roots: WorkspaceRoots,
):
    first = _candidate(workspace_roots, "aaaa_v1")
    second = _candidate(workspace_roots, "bbbb_v2")
    naive = _candidate(workspace_roots, "cccc_v3")

    promote(workspace_roots, SLOT, first)
    assert promote(workspace_roots, SLOT, naive, profile="naive") is None
    assert promote(workspace_roots, SLOT, second) == first

    assert promoted_reference(workspace_roots, SLOT) == second
    assert promoted_reference(workspace_roots, SLOT, profile="naive") == naive
    written = pointer_path(workspace_roots, Kind.TOKENIZER).read_text(encoding="utf-8")
    assert json.loads(written) == {
        "naive": {SLOT.uri: {"reference": naive.uri, "replaced": None}},
        "promoted": {SLOT.uri: {"reference": second.uri, "replaced": first.uri}},
    }


def test_nothing_promoted_is_refused_naming_the_selection(
    workspace_roots: WorkspaceRoots,
):
    with pytest.raises(NothingPromotedError, match="tokenizer://ie/wordpiece"):
        promoted_reference(workspace_roots, SLOT)


def test_a_profile_naming_nothing_is_refused_by_name_and_a_malformed_one_outright(
    workspace_roots: WorkspaceRoots,
):
    promote(workspace_roots, SLOT, _candidate(workspace_roots, "aaaa_v1"))

    with pytest.raises(NothingPromotedError, match="no naive reference"):
        promoted_reference(workspace_roots, SLOT, profile="naive")
    with pytest.raises(ValueError, match="short lowercase name"):
        promoted_reference(workspace_roots, SLOT, profile="Not A Name")


def test_an_absent_candidate_is_refused_naming_its_reference(
    workspace_roots: WorkspaceRoots,
):
    """A checkout holds the tracked pointer before its artifact store holds the
    candidate, so the failure has to say which one to train."""
    candidate = _candidate(workspace_roots, "aaaa_v1")
    promote(workspace_roots, SLOT, candidate)
    locate(workspace_roots, candidate).rmdir()

    with pytest.raises(
        PointerError, match="tokenizer://ie/wordpiece/aaaa_v1"
    ) as raised:
        promoted_reference(workspace_roots, SLOT)
    assert not isinstance(raised.value, NothingPromotedError)


def test_promotion_refuses_what_the_selection_does_not_name_or_cannot_load(
    workspace_roots: WorkspaceRoots,
):
    elsewhere = _candidate(workspace_roots, "aaaa_v1", scope="gb")
    absent = reference(
        Kind.TOKENIZER, Side.DATA, scope="ie", tokenizer_id="wordpiece", key="cccc_v3"
    )

    with pytest.raises(PointerError, match="does not name"):
        promote(workspace_roots, SLOT, elsewhere)
    with pytest.raises(PointerError, match="absent"):
        promote(workspace_roots, SLOT, absent)
    with pytest.raises(PointerError, match="does not name"):
        promote(workspace_roots, SLOT, SLOT)
    assert not pointer_path(workspace_roots, Kind.TOKENIZER).exists()
