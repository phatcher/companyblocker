"""The promoted tokenizer a test lays out by hand.

A reader finds a tokenizer through `config/tokenizers.json`, so a test that
needs one on disk gets a candidate directory, marked finished, and the pointer
naming it, then writes or trains the model at the path it is handed. A test of
promotion itself goes through `training.promotion` instead.
"""

from __future__ import annotations

from company_tokenize.paths import (
    TokenizerDirectoryFiles,
    tokenizer_directory_files,
    tokenizer_id,
)

from tests.production_records import write_record
from workspace.kind_layout import Kind
from workspace.pointer import PROMOTED_PROFILE, promote
from workspace.reference import Side, locate, reference
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import tokenizer_selection


def promoted_tokenizer_files(
    roots: WorkspaceRoots,
    *,
    system: str | None,
    trainer: str = "wordpiece",
    tokenizer_encoding: str | None = None,
    key: str = "fixture",
    profile: str = PROMOTED_PROFILE,
) -> TokenizerDirectoryFiles:
    """Point `profile`, `promoted` unless given, at an empty candidate for a
    scope and trainer and name its files."""
    selection = tokenizer_selection(
        system=system, tokenizer_id=tokenizer_id(trainer, tokenizer_encoding)
    )
    candidate = reference(Kind.TOKENIZER, Side.DATA, **selection.fields, key=key)
    directory = locate(roots, candidate)
    write_record(roots, directory)
    promote(roots, selection, candidate, profile=profile)
    return tokenizer_directory_files(directory, trainer=trainer)
