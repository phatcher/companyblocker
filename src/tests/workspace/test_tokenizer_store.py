"""Direct tests for `workspace.tokenizer_store`."""

from __future__ import annotations

import pytest

from workspace.kind_layout import Kind
from workspace.pointer import PointerError, promote
from workspace.reference import Side, locate, reference
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import (
    promoted_tokenizer,
    promoted_tokenizer_directory,
    scope_system,
    tokenizer_selection,
)


def test_a_scope_is_a_lowercased_system_code_or_global():
    assert tokenizer_selection(system=" IE ", tokenizer_id="wordpiece").uri == (
        "tokenizer://ie/wordpiece"
    )
    assert tokenizer_selection(system=None, tokenizer_id="sentencepiece_bpe").uri == (
        "tokenizer://global/sentencepiece_bpe"
    )


def test_the_promoted_directory_is_the_pointed_candidate(
    workspace_roots: WorkspaceRoots,
):
    candidate = reference(
        Kind.TOKENIZER, Side.DATA, scope="ie", tokenizer_id="wordpiece", key="aaaa_v1"
    )
    locate(workspace_roots, candidate).mkdir(parents=True)
    promote(
        workspace_roots,
        tokenizer_selection(system="ie", tokenizer_id="wordpiece"),
        candidate,
    )

    assert promoted_tokenizer(
        workspace_roots, system="ie", tokenizer_id="wordpiece"
    ) == (candidate)
    assert promoted_tokenizer_directory(
        workspace_roots, system="IE", tokenizer_id="wordpiece"
    ) == locate(workspace_roots, candidate)


def test_an_unpromoted_scope_is_refused(workspace_roots: WorkspaceRoots):
    with pytest.raises(PointerError, match="tokenizer://gb/wordpiece"):
        promoted_tokenizer_directory(
            workspace_roots, system="gb", tokenizer_id="wordpiece"
        )


def test_a_country_scope_needs_its_system_and_a_global_scope_drops_it():
    assert scope_system(" Country ", "ie") == "ie"
    assert scope_system("global", "ie") is None
    with pytest.raises(ValueError, match="system code is required"):
        scope_system("country", None)
    with pytest.raises(ValueError, match="Unsupported tokenizer scope"):
        scope_system("team", "ie")
