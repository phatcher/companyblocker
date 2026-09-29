"""Jurisdiction-scoped checkpoint selection, generalised over any registry.

`company_vectorize.clustering_policy.resolve_sbert_model_for_jurisdictions` picks one
registered sentence-embedding checkpoint for a run's countries, with three fallbacks to a
stated default: the countries are unset, they span more than one monolingual entry's
coverage (ambiguous), or they match no registered monolingual entry at all. That rule is not
sbert-specific -- it only needs a registry of entries each declaring a `coverage` and a
`jurisdictions` tuple, plus a way to read the identifier a match resolves to. This module
holds that rule for this package, parameterised by the registry and the identifier reader, so
every per-language checkpoint registry here (`fasttext_registry.py`'s pretrained fastText
checkpoints, and a domain-trained registry added later) selects through the same function. It
mirrors `resolve_sbert_model_for_jurisdictions` rather than calling it, because this package
does not depend on `company_vectorize`; a change to the rule is made in both.

A multilingual entry is never returned: it exists for an explicit override, not as an
inferred per-jurisdiction default, matching `resolve_sbert_model_for_jurisdictions`'s own
rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol


class _JurisdictionScopedEntry(Protocol):
    """The two fields the selection rule reads off a registry entry.

    Structural, not a base class: `SbertModelEntry`, `FastTextCheckpointEntry`, and any
    future registry entry dataclass satisfy this without inheriting from anything here.
    Declared as read-only properties, not plain attributes, so a frozen dataclass's fields
    (also read-only) satisfy this Protocol -- mypy treats a plain-attribute Protocol member
    as requiring a settable field on the implementer, which no frozen dataclass has.
    """

    @property
    def coverage(self) -> str: ...

    @property
    def jurisdictions(self) -> tuple[str, ...]: ...


def resolve_checkpoint_for_jurisdictions[EntryT: _JurisdictionScopedEntry](
    jurisdictions: tuple[str, ...] | None,
    registry: Sequence[EntryT],
    *,
    identifier: Callable[[EntryT], str],
    default: str,
) -> str:
    """Pick one registered checkpoint's identifier for a run scoped to `jurisdictions`.

    Selection is per run rather than per record: the caller supplies the jurisdictions its
    run is already scoped to, and this picks the one registered *monolingual* entry whose own
    `jurisdictions` covers all of them -- on the view that a run scoped to a single
    jurisdiction supplies real language signal for free.

    Falls back to `default` whenever `jurisdictions` is `None`/empty, spans more than one
    monolingual entry's coverage (ambiguous), or matches no registered monolingual entry at
    all, since most runs are not language-specific and an unmatched jurisdiction should
    resolve to something usable rather than fail the run.

    Args:
        jurisdictions: `jurisdiction_code` values the run is scoped to (for example
            `BlockingRunConfig.countries`), or `None`/empty for no scoping.
        registry: Every candidate entry, in any order.
        identifier: Reads the value to return from a matched entry -- a checkpoint string for
            `SbertModelEntry`, a slug for `FastTextCheckpointEntry`.
        default: Returned whenever no single monolingual entry matches.
    """
    if jurisdictions:
        wanted = {str(value).strip().lower() for value in jurisdictions}
        candidates = [
            entry
            for entry in registry
            if entry.coverage == "monolingual" and wanted <= set(entry.jurisdictions)
        ]
        if len(candidates) == 1:
            return identifier(candidates[0])
    return default
