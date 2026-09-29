"""Publish a trained tokenizer in two acts: store the candidate, then point at it.

Training has written the model into the scope's working directory by the time
this runs. What is added here is the part a reader depends on. Storing copies
the candidate whole into the location its `tokenizer://` reference names,
through `workspace.records.produce` so it carries a record of what made it.
Pointing moves a profile name in `config/tokenizers.json` onto a stored
candidate, keeping the reference it replaced. A promotion is the two in order;
either is also done alone, since a candidate can be stored without being
promoted and a stored one promoted without being trained again.
The directory's contents and its key are `company_tokenize`'s; where it sits,
its record and the pointer are `workspace`'s.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from company_tokenize import (
    compute_corpus_content_hash,
    promoted_candidate_key,
    tokenizer_id,
    write_candidate_directory,
)

from workspace.kind_layout import Kind
from workspace.pointer import PROMOTED_PROFILE, promote
from workspace.records import produce, read_record
from workspace.reference import Reference, Side, reference, select_references
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import tokenizer_selection

NAIVE_PROFILE = "naive"


def store_candidate(
    *,
    roots: WorkspaceRoots,
    system: str | None,
    trainer: str,
    tokenizer_encoding: str | None,
    model_path: Path,
    metadata_path: Path,
    token_scores_path: Path | None,
    corpus_path: Path,
    vocab_size: int | None,
    min_frequency: int,
    metrics: Mapping[str, object],
    optimize_summary: Mapping[str, object] | None = None,
    parameters: Mapping[str, object],
    invocation: Sequence[str],
) -> Reference:
    """Write a trained tokenizer into the store under its reference, with a
    record of what made it, and return that reference. No profile is moved.

    Where the files were trained is not this function's concern: it copies
    what it is handed. The corpus is not yet a reference, so its content hash
    is recorded as a parameter rather than as an input. `metrics` is what the
    candidate measured against its training corpus and `optimize_summary` the
    evidence of the optimize run that chose it, when one did. Both are written into
    the candidate's directory here because it cannot be added to afterwards.
    """
    corpus_content_hash = compute_corpus_content_hash(corpus_path)
    selection = tokenizer_selection(
        system=system, tokenizer_id=tokenizer_id(trainer, tokenizer_encoding)
    )
    key = promoted_candidate_key(
        corpus_content_hash=corpus_content_hash,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        model_path=model_path,
    )
    candidate = reference(Kind.TOKENIZER, Side.DATA, **selection.fields, key=key)
    with produce(
        roots,
        candidate,
        inputs={},
        parameters={
            **parameters,
            "corpus_content_hash": corpus_content_hash,
            "vocab_size": None if vocab_size is None else int(vocab_size),
            "min_frequency": int(min_frequency),
        },
        invocation=invocation,
        key=key,
    ) as production:
        write_candidate_directory(
            production.directory,
            model=model_path,
            metadata=metadata_path,
            token_scores=token_scores_path,
            metrics=metrics,
            optimize_summary=optimize_summary,
            trainer=trainer,
        )
    return candidate


def find_stored_candidate(
    *,
    roots: WorkspaceRoots,
    system: str | None,
    trainer: str,
    tokenizer_encoding: str | None,
    corpus_path: Path,
    settings: Mapping[str, object],
) -> Reference | None:
    """The stored candidate trained on this corpus with these settings, or None.

    Training the same settings twice gives two slightly different models, so a
    caller that wants one tokenizer per corpus and settings asks here first
    and trains only when nothing is stored. A candidate matches when its
    record carries the corpus's content hash and every one of `settings`,
    compared as the record's JSON holds them.
    """
    wanted = {
        **json.loads(json.dumps(dict(settings))),
        "corpus_content_hash": compute_corpus_content_hash(corpus_path),
    }
    selection = tokenizer_selection(
        system=system, tokenizer_id=tokenizer_id(trainer, tokenizer_encoding)
    )
    for candidate in select_references(roots, selection):
        record = read_record(roots, candidate)
        if record is not None and all(
            record.parameters.get(name) == value for name, value in wanted.items()
        ):
            return candidate
    return None


def point_profile_at_candidate(
    roots: WorkspaceRoots, candidate: Reference, *, profile: str = PROMOTED_PROFILE
) -> Reference | None:
    """Make a stored candidate the one `profile` names for its scope and
    tokenizer id, and return the reference it replaced, or None.

    The candidate is already in the store, so nothing is trained or copied.
    """
    slot = {name: value for name, value in candidate.values if name != "key"}
    selection = reference(candidate.kind, candidate.side, **slot)
    return promote(roots, selection, candidate, profile=profile)


def publish_promoted_candidate(
    *, roots: WorkspaceRoots, **trained: Any
) -> tuple[Reference, Reference | None]:
    """Store the candidate, then point `promoted` at it: what the optimiser's
    final step does with the candidate it chose.

    Takes `store_candidate`'s arguments. Returns the candidate's reference and
    the one `promoted` named before, or None.
    """
    candidate = store_candidate(roots=roots, **trained)
    return candidate, point_profile_at_candidate(roots, candidate)


def publish_naive_candidate(
    *, roots: WorkspaceRoots, **trained: Any
) -> tuple[Reference, Reference | None]:
    """Store the candidate, then point `naive` at it: what plain training does
    with its result, a first guess at the settings and not a choice among
    candidates, so `promoted` is left where it was.

    Takes `store_candidate`'s arguments. Returns the candidate's reference and
    the one `naive` named before, or None.
    """
    candidate = store_candidate(roots=roots, **trained)
    return candidate, point_profile_at_candidate(
        roots, candidate, profile=NAIVE_PROFILE
    )


__all__ = [
    "NAIVE_PROFILE",
    "find_stored_candidate",
    "point_profile_at_candidate",
    "publish_naive_candidate",
    "publish_promoted_candidate",
    "store_candidate",
]
