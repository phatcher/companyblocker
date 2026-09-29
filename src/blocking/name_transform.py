"""Both sides' name forms derived live, and one named function chosen to compare.

Two things happen to a name before a run compares it, and they are kept
apart because they answer different questions.

The first is unconditional. Every run derives both sides' name forms live,
through `company_cleanse.cleanse_lazyframe` under the run's own named
normalization profile (`derive_name_forms`). The loaded layer's own
`name_cleansed`, `name_cleansed_basic`, `short_name` and `acronym` are
overwritten, never read: a materialized column reflects whichever ruleset
was active when that system's Cleanse stage last ran, two systems cleansed
at different times disagree on the same name, and a perturbed row never
passed the Cleanse stage at all. So the forms a run scores, and the forms
its name-equality levels are classified from, are the forms this run
derived under one profile on both sides.

The second is a choice. The transform names which of those forms both sides
are compared on (`NameTransform`): `cleanse`, the default, scores the cleansed
name, so the scan compares the text the run's cleanse profile shapes and the
exact-name joins already compare; `acronym` and `short_name` score the derived
column of that name; `identity` scores the raw `name`. A run compares two names, and the comparison is only meaningful
when both sides reached it the same way, so the transform is one function
applied to both sides rather than a bespoke asymmetric matcher deriving a
key on one side and hoping the other already has one. What is shared is the
*application*; the similarity measure is untouched.

The transform sits before, and composes with, `BlockingStrategyConfig.name_source`,
the source-only column choice: under `identity` that choice is honoured as
it always was, and under any other transform both sides score the
transform's own column, so `name_source` must be `name`.

Every transform is a frozen dataclass registered by name in
`NAME_TRANSFORMS`, so `resolve_name_transform` is the one place a name
becomes a function and a new function is registered rather than wired in. A
transform that derives something the cleanse does not can do so in `apply`;
the cleanse-derived ones need nothing there, since the forms already exist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

# The derivation itself is shared with `validation`, which scores the same
# names and must reach the same stored forms; this module chooses which of
# them a run compares.
from validation.name_forms import (
    NAME_FORM_COLUMNS,
    RAW_NAME_COLUMN,
    derive_name_forms,
    name_forms_ruleset_digest,
)

DEFAULT_CLEANSE_PROFILE = "default"
"""`CleanseConfig.normalization_profile` a run derives name forms under
unless told otherwise: the package's own built-in chain."""


@runtime_checkable
class NameTransform(Protocol):
    """One function applied identically to both sides of a run."""

    @property
    def kind(self) -> str:
        """The registered name, part of the run identity."""
        ...

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        """`frame` with anything this transform derives beyond the cleanse
        forms `derive_name_forms` already put on it."""
        ...

    def scored_column(self, name_col: str) -> str:
        """The column the run scores on, given the column it would otherwise
        have read (`strategy.name_source` on the source side, `name` on the
        target)."""
        ...

    def describe(self) -> dict[str, object]:
        """What a run applied, as recorded on its own output."""
        ...


@dataclass(slots=True, frozen=True)
class IdentityTransform:
    """Score the raw name: each side compares the column it always read."""

    @property
    def kind(self) -> str:
        return "identity"

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    def scored_column(self, name_col: str) -> str:
        return name_col

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "column": RAW_NAME_COLUMN}


@dataclass(slots=True, frozen=True)
class DerivedColumnTransform:
    """Score one of the cleanse-derived forms on both sides.

    `column` is one `cleanse_lazyframe` derives: `name_cleansed` for the
    cleansed name, `short_name` for the suffix and noise-word stem, `acronym`
    for the initials candidate. A row whose derived value is null (an
    `acronym` for most names) contributes no candidates, the same way a null
    `short_name` already does under `name_source="short_name"`.
    """

    kind: str
    column: str

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.column not in frame.columns:
            raise ValueError(
                f"name transform {self.kind!r} scores {self.column!r}, which the "
                "frame does not carry: derive_name_forms must run first"
            )
        return frame

    def scored_column(self, name_col: str) -> str:
        return self.column

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "column": self.column}


NAME_TRANSFORMS: Mapping[str, Callable[[], NameTransform]] = {
    "identity": IdentityTransform,
    "cleanse": lambda: DerivedColumnTransform("cleanse", "name_cleansed"),
    "acronym": lambda: DerivedColumnTransform("acronym", "acronym"),
    "short_name": lambda: DerivedColumnTransform("short_name", "short_name"),
}
"""Every transform a run can select, by the name a run selects it with. A
new function is one more entry here."""

IDENTITY_NAME_TRANSFORM = "identity"
"""The one transform that scores the raw name, and so the one under which the
source-only `name_source` column choice still applies."""

# The cleansed name: what the run's cleanse profile shapes, and the form the
# exact-name joins already compare, so the scan and the joins see one text.
DEFAULT_NAME_TRANSFORM = "cleanse"


def resolve_name_transform(name: str) -> NameTransform:
    """The transform a run's `strategy.name_transform` names."""
    key = name.strip().lower()
    try:
        factory = NAME_TRANSFORMS[key]
    except KeyError:
        allowed = ", ".join(sorted(NAME_TRANSFORMS))
        raise ValueError(f"name_transform must be one of: {allowed}") from None
    return factory()


__all__ = [
    "DEFAULT_CLEANSE_PROFILE",
    "DEFAULT_NAME_TRANSFORM",
    "IDENTITY_NAME_TRANSFORM",
    "NAME_FORM_COLUMNS",
    "NAME_TRANSFORMS",
    "RAW_NAME_COLUMN",
    "DerivedColumnTransform",
    "IdentityTransform",
    "NameTransform",
    "derive_name_forms",
    "name_forms_ruleset_digest",
    "resolve_name_transform",
]
