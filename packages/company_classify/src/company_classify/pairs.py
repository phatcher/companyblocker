"""Self-supervised pair-generation contract and its validation checks.

Every pair a producer emits (`pair_producers.py`) is one `PairRecord`: two names, a match /
non-match label, the `source` tag naming which producer made it, the two entities the names came
from, and the `split` those entities belong to under the one shared `EntitySplitter`. The
`source` tag and the split tag together are what makes a later synthetic-versus-real comparison
possible without a bespoke holdout: real-alias pairs whose entity landed in `test` were never
trainable, whichever producer supplied the training pairs.

The contract is enforced rather than assumed. `validate_pairs()` runs three checks:

- Structural compliance (`find_contract_violations()`): labels are binary, names and entity URIs
  are populated, a match pair names one entity twice, a non-match pair names two different
  entities, and the `split` tag is the one the shared splitter actually computes for those
  entities. The last check is the load-bearing one: it fails a producer that computed its own
  split instead of tagging against the shared one.
- Split leakage (`find_split_leakage()`): one entity may appear in only one partition. A single
  entity contributing pairs to both `train` and `test` means a name variant of something trained
  on is also being evaluated against.
- Ambiguity (`find_ambiguous_pairs()`): the same two names labelled both match and non-match
  anywhere in the set, and pairs whose two names are identical. Both are label noise a
  downstream classifier cannot resolve, and both are cheap to detect here rather than to
  diagnose later as an unexplained accuracy ceiling.

Producer inputs are two small shapes rather than a repo-wide schema: `EntityRecord` (one row per
entity: the name to perturb, and the `short_name` already derived by `company_cleanse` when
present) and `NameVariant` (one row per observed name variant, the shape of the canonical-stage
`*-names-*.parquet` sidecars).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .entity_split import EntitySplitter, SplitName

MATCH_LABEL = 1
NON_MATCH_LABEL = 0


class PairSource(StrEnum):
    """Which producer generated a pair."""

    SYNTHETIC = "synthetic"
    REAL_ALIAS = "real_alias"
    NAME_SHORT_NAME = "name_short_name"


class PairContractError(ValueError):
    """Raised by `validate_pairs()` when a pair set breaks the contract above."""


@dataclass(frozen=True)
class PairRecord:
    """One generated training pair.

    Attributes:
        left_name: The anchor name of the pair.
        right_name: The counterpart name: a variant of `left_name`'s entity for a match pair,
            or a different entity's name for a non-match pair.
        label: `MATCH_LABEL` (1) or `NON_MATCH_LABEL` (0).
        source: The producer that generated this pair.
        split: Partition both entities belong to under the shared `EntitySplitter`.
        left_system_uri: Entity `left_name` came from.
        right_system_uri: Entity `right_name` came from. Equal to `left_system_uri` for a match
            pair.
        detail: Optional producer-specific provenance (a perturbation scenario id, the
            `name_type` a real variant came from). Carried through for analysis; nothing in this
            package reads it.
    """

    left_name: str
    right_name: str
    label: int
    source: PairSource
    split: SplitName
    left_system_uri: str
    right_system_uri: str
    detail: str | None = None

    @property
    def is_match(self) -> bool:
        return self.label == MATCH_LABEL


@dataclass(frozen=True)
class EntityRecord:
    """One entity as a pair producer consumes it.

    Attributes:
        system_uri: Entity identity, and the sole input to split membership.
        name: The entity's primary name.
        system: Source system code, used by the synthetic producer's scenario exclusions.
        country: Optional country code, used by the same exclusions and by country-aware
            perturbation operators.
        short_name: The reduction `company_cleanse` derives, when it is available. Only the
            name/short_name producer reads it.
    """

    system_uri: str
    name: str
    system: str = ""
    country: str | None = None
    short_name: str | None = None


@dataclass(frozen=True)
class NameVariant:
    """One observed name variant of an entity, as carried by the `*-names-*.parquet` sidecars.

    Attributes:
        system_uri: The entity this variant belongs to -- an opaque identity supplied by the
            caller, and the sole input to grouping and split membership. This package reads it
            straight through; a `*-names-*.parquet` sidecar's own `system_uri` column is often
            the *row's* own id, not its entity's (a name variant's is one per row), so a caller
            reading such a sidecar resolves it to the entity first (e.g.
            `workspace.match_resolution.entity_of`, which walks a name row or a perturbed row
            back to its entity) before building a `NameVariant`.
        name: The variant name text.
        name_type: The sidecar's own variant classification (`primary`, `previous`, `official`,
            `short`, `alias`, `label`, and similar). Drives anchor selection in the real-alias
            producer and is carried into `PairRecord.detail`.
    """

    system_uri: str
    name: str
    name_type: str | None = None


def name_variants_from_rows(rows: Iterable[dict[str, object]]) -> list[NameVariant]:
    """Adapt `*-names-*.parquet` rows (e.g. `DataFrame.iter_rows(named=True)`) to `NameVariant`.

    Only `system_uri`, `name`, and `name_type` are read; the sidecars' remaining columns
    (`source_type`, `language_code`, `derivation_note`, the system's own id column) are ignored.
    `system_uri` is carried through exactly as given, with no resolution of its own: the caller
    supplies whatever opaque identity a `NameVariant` should be grouped and split by (see
    `NameVariant.system_uri`).
    """
    variants: list[NameVariant] = []
    for row in rows:
        name_type = row.get("name_type")
        variants.append(
            NameVariant(
                system_uri=str(row["system_uri"]),
                name=str(row["name"]),
                name_type=None if name_type is None else str(name_type),
            )
        )
    return variants


def _split_lookup(
    pairs: Sequence[PairRecord], splitter: EntitySplitter
) -> dict[str, SplitName]:
    entities = sorted(
        {pair.left_system_uri for pair in pairs if _is_populated(pair.left_system_uri)}
        | {
            pair.right_system_uri
            for pair in pairs
            if _is_populated(pair.right_system_uri)
        }
    )
    return dict(zip(entities, splitter.assign_many(entities), strict=True))


def _is_populated(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def find_contract_violations(
    pairs: Sequence[PairRecord], *, splitter: EntitySplitter
) -> list[str]:
    """Report every structural contract breach in `pairs`, as one message per breach."""
    if not pairs:
        return []

    expected_split = _split_lookup(pairs, splitter)
    violations: list[str] = []

    for index, pair in enumerate(pairs):
        prefix = f"pair[{index}] ({pair.source.value})"

        if pair.label not in (MATCH_LABEL, NON_MATCH_LABEL):
            violations.append(f"{prefix}: label {pair.label!r} is not 0 or 1")
        if not _is_populated(pair.left_name) or not _is_populated(pair.right_name):
            violations.append(f"{prefix}: both pair members must be non-empty names")
        if not _is_populated(pair.left_system_uri) or not _is_populated(
            pair.right_system_uri
        ):
            violations.append(f"{prefix}: both pair members must carry a system_uri")
            continue

        same_entity = pair.left_system_uri == pair.right_system_uri
        if pair.label == MATCH_LABEL and not same_entity:
            violations.append(
                f"{prefix}: match pair spans two entities "
                f"({pair.left_system_uri} != {pair.right_system_uri})"
            )
        if pair.label == NON_MATCH_LABEL and same_entity:
            violations.append(
                f"{prefix}: non-match pair names one entity twice "
                f"({pair.left_system_uri})"
            )

        for side, system_uri in (
            ("left", pair.left_system_uri),
            ("right", pair.right_system_uri),
        ):
            if pair.split != expected_split[system_uri]:
                violations.append(
                    f"{prefix}: split {pair.split.value!r} disagrees with the shared "
                    f"splitter, which puts {side} entity {system_uri} in "
                    f"{expected_split[system_uri].value!r}"
                )

    return violations


def find_split_leakage(pairs: Sequence[PairRecord]) -> dict[str, set[SplitName]]:
    """Report entities tagged with more than one partition, entity to partitions seen."""
    seen: dict[str, set[SplitName]] = {}
    for pair in pairs:
        for system_uri in (pair.left_system_uri, pair.right_system_uri):
            seen.setdefault(system_uri, set()).add(pair.split)
    return {
        system_uri: splits for system_uri, splits in seen.items() if len(splits) > 1
    }


def find_ambiguous_pairs(pairs: Sequence[PairRecord]) -> list[str]:
    """Report unlearnable pairs: conflicting labels for one name pair, and self-pairs."""
    labels_by_names: dict[tuple[str, str], set[int]] = {}
    ambiguities: list[str] = []

    for index, pair in enumerate(pairs):
        if pair.left_name == pair.right_name:
            ambiguities.append(
                f"pair[{index}] ({pair.source.value}): both members are the same text "
                f"{pair.left_name!r}"
            )
        first_name, second_name = sorted((pair.left_name, pair.right_name))
        labels_by_names.setdefault((first_name, second_name), set()).add(pair.label)

    for (left, right), labels in sorted(labels_by_names.items()):
        if len(labels) > 1:
            ambiguities.append(
                f"names ({left!r}, {right!r}) are labelled both match and non-match"
            )

    return ambiguities


def validate_pairs(pairs: Sequence[PairRecord], *, splitter: EntitySplitter) -> None:
    """Run every contract check over `pairs`, raising `PairContractError` on any finding."""
    problems = find_contract_violations(pairs, splitter=splitter)
    problems.extend(
        f"entity {system_uri} appears in partitions "
        f"{sorted(split.value for split in splits)}"
        for system_uri, splits in sorted(find_split_leakage(pairs).items())
    )
    problems.extend(find_ambiguous_pairs(pairs))

    if problems:
        listed = "\n".join(f"  - {problem}" for problem in problems)
        raise PairContractError(
            f"{len(problems)} pair-contract problem(s) found:\n{listed}"
        )
