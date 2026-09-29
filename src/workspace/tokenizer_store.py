"""Where the promoted tokenizer for a scope and tokenizer id sits.

A reader names a scope, a system code or none for the global one, and a tokenizer
id as `company_tokenize` spells it, and is handed the promoted candidate's
directory: the pointer in `config/tokenizers.json` says which candidate, and the
`tokenizer://` layout says where it is. What the directory holds is
`company_tokenize`'s, so a reader asks that package for a file inside it.
"""

from __future__ import annotations

from pathlib import Path

from .artifact_layout import GLOBAL_TOKENIZER_SCOPE
from .kind_layout import Kind
from .pointer import PROMOTED_PROFILE, promoted_reference
from .reference import Reference, Side, locate, reference
from .roots import WorkspaceRoots


def tokenizer_selection(*, system: str | None, tokenizer_id: str) -> Reference:
    """The pointer slot one scope and tokenizer id share: `tokenizer://<scope>/<tokenizer_id>`."""
    return reference(
        Kind.TOKENIZER,
        Side.DATA,
        scope=GLOBAL_TOKENIZER_SCOPE if system is None else system.strip().lower(),
        tokenizer_id=tokenizer_id,
    )


def scope_system(scope: str, system: str | None) -> str | None:
    """The system a tokenizer scope names: `system` for the `country` scope,
    which requires one, and none for `global`."""
    normalized = scope.strip().lower()
    if normalized == "global":
        return None
    if normalized != "country":
        raise ValueError(
            f"Unsupported tokenizer scope '{scope}'. Expected 'country' or 'global'."
        )
    if system is None or not system.strip():
        raise ValueError("A system code is required for the country tokenizer scope.")
    return system


def promoted_tokenizer(
    roots: WorkspaceRoots,
    *,
    system: str | None,
    tokenizer_id: str,
    profile: str = PROMOTED_PROFILE,
) -> Reference:
    """The candidate `profile` names for a scope and tokenizer id, the promoted
    one unless another profile is given; raises `pointer.PointerError` when it
    names none, or when the candidate is absent on disk."""
    return promoted_reference(
        roots,
        tokenizer_selection(system=system, tokenizer_id=tokenizer_id),
        profile=profile,
    )


def promoted_tokenizer_directory(
    roots: WorkspaceRoots,
    *,
    system: str | None,
    tokenizer_id: str,
    profile: str = PROMOTED_PROFILE,
) -> Path:
    """The directory of the candidate `profile` names for a scope and tokenizer id."""
    return locate(
        roots,
        promoted_tokenizer(
            roots, system=system, tokenizer_id=tokenizer_id, profile=profile
        ),
    )


__all__ = [
    "promoted_tokenizer",
    "promoted_tokenizer_directory",
    "scope_system",
    "tokenizer_selection",
]
