"""The reference a command line's natural facts name.

A person gives a source system, a target system, a profile, a seed or a
benchmark's name, and is shown the same; nobody types a URI or a layout. This
module is the one place those facts become a reference, so no script composes
one: `source_request` for what a run reads as its source, `target_request` for
what it scores against, `profile_request` for the recipe a materializer applies.

What comes back is a request where a version, a date or a seed was left out:
`reference.resolve_latest` pins it to what is on disk before anything reads,
keys or records it.
"""

from __future__ import annotations

from .derived_uri import PERTURBED_SCHEME
from .kind_layout import Kind
from .reference import (
    CANONICAL_DATASET,
    MATCHED_DATASET,
    NAMES_DATASET,
    URI_SEPARATOR,
    InvalidReferenceError,
    Reference,
    Side,
    parse_reference,
    reference,
)

ENTITY_TRUTH_COLUMN = "match_uri"
DERIVED_TRUTH_COLUMN = "source_uri"


def _given(**values: object) -> dict[str, str]:
    return {name: str(value) for name, value in values.items() if value is not None}


def source_request(
    *,
    source: str | None = None,
    dataset: str | None = None,
    perturbed: str | None = None,
    version: str | None = None,
    seed: int | None = None,
    date: str | None = None,
    benchmark: str | None = None,
) -> Reference:
    """What a run reads as its source.

    A system alone is its labelled rows, `matched`. `dataset="names"` reads the
    system's names in place of its entities, and `perturbed` reads either as
    perturbed under that profile; the two compose. A version or a seed belongs
    to a perturbed source and a date to a dated one, and each may be left out. A
    benchmark supplies both sides at once and is no system, so it stands alone.
    """
    if benchmark is not None:
        beside = _given(source=source, dataset=dataset, perturbed=perturbed)
        if beside:
            raise InvalidReferenceError(
                f"A benchmark stands in place of a source and a target, so it is "
                f"refused beside {sorted(beside)}."
            )
        raise InvalidReferenceError(
            f"Benchmark {benchmark!r} has no registered dataset: its tables are "
            "unpacked from an archive and not yet written as a production."
        )
    if source is None:
        raise InvalidReferenceError("A source names a system or a benchmark.")
    if dataset not in (None, NAMES_DATASET):
        raise InvalidReferenceError(
            f"{dataset!r} is no dataset to read in place of a system's entities; "
            f"the one such dataset is {NAMES_DATASET!r}."
        )
    names = _given(dataset=dataset)
    if perturbed is None:
        stray = _given(version=version, seed=seed)
        if stray:
            raise InvalidReferenceError(
                f"{sorted(stray)} belong to a perturbed source, and none is named."
            )
        if dataset is None:
            if date is not None:
                raise InvalidReferenceError(
                    f"{source}'s {MATCHED_DATASET} rows have no date to pin."
                )
            return reference(
                Kind.DATA, Side.DATA, system=source, dataset=MATCHED_DATASET
            )
        return reference(
            Kind.DATA, Side.DATA, system=source, **names, **_given(version=date)
        )
    if date is not None:
        raise InvalidReferenceError(
            "A perturbed dataset pins the snapshot it was made from in its record, "
            "so a run over one takes no date."
        )
    return reference(
        Kind.PERTURBATION,
        Side.DATA,
        source=source,
        profile=perturbed,
        **names,
        **_given(version=version, seed=seed),
    )


def target_request(*, target: str, date: str | None = None) -> Reference:
    """What a run scores against: a system's `canonical` snapshot, the latest
    unless dated."""
    return reference(
        Kind.DATA,
        Side.DATA,
        system=target,
        dataset=CANONICAL_DATASET,
        **_given(version=date),
    )


def profile_request(
    *, perturb: str, version: str | None = None, draft: bool = False
) -> Reference:
    """The perturbation profile a materializer applies: the latest promoted
    version unless one is named, or its draft."""
    stage = {"stage": "draft"} if draft else {}
    return reference(
        Kind.PERTURBATION,
        Side.PROFILE,
        name=perturb,
        **_given(version=version),
        **stage,
    )


def perturbed_source(source: str) -> Reference | None:
    """The perturbed dataset a source names, or None where it is a system code.

    A run's configuration carries its source as one string: a system code, or a
    perturbed dataset's own URI, `perturbed://ie/en-lite/v1/42`. There is no
    second spelling, and this is the one place the string is read.
    """
    if not source.startswith(f"{PERTURBED_SCHEME}{URI_SEPARATOR}"):
        return None
    dataset = parse_reference(source)
    if not dataset.complete:
        raise InvalidReferenceError(f"{source} names no one perturbed dataset.")
    return dataset


def is_perturbed_source(source: str) -> bool:
    return perturbed_source(source) is not None


def perturbed_parts(source: str) -> tuple[str, str, str, int]:
    """`(source_system, profile_id, version, seed)` of a perturbed source."""
    dataset = perturbed_source(source)
    if dataset is None:
        raise InvalidReferenceError(f"{source!r} is no perturbed dataset.")
    fields = dataset.fields
    return fields["source"], fields["profile"], fields["version"], int(fields["seed"])


def perturbed_source_uri(
    *, source_system: str, profile_id: str, version: str, seed: int
) -> str:
    """The string a run's configuration carries for one perturbed dataset."""
    return source_request(
        source=source_system, perturbed=profile_id, version=version, seed=seed
    ).uri


def implied_truth_column(source: Reference) -> str:
    """The column a source's truth sits in. A system's own rows carry their
    match; names and perturbed rows are derived, so theirs is the row they came
    from, and nobody types it."""
    derived = (
        source.kind is Kind.PERTURBATION
        or source.fields.get("dataset") == NAMES_DATASET
    )
    return DERIVED_TRUTH_COLUMN if derived else ENTITY_TRUTH_COLUMN


__all__ = [
    "DERIVED_TRUTH_COLUMN",
    "ENTITY_TRUTH_COLUMN",
    "implied_truth_column",
    "is_perturbed_source",
    "perturbed_parts",
    "perturbed_source",
    "perturbed_source_uri",
    "profile_request",
    "source_request",
    "target_request",
]
