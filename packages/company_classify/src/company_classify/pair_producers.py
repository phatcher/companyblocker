"""Three interchangeable pair producers for self-supervised training.

Each producer answers the same question from a different source of name variation, and all three
emit the same `PairRecord` shape (`pairs.py`) tagged against the same `EntitySplitter`
(`entity_split.py`), so they can be combined, swapped, or compared without anything downstream
knowing which one produced a given pair:

- `SyntheticPairProducer`: generated variation. A perturbed name paired with the name it came
  from. The operator set is `company_perturbation`'s profile contract, called through
  `perturb_records()`; no perturbation logic is defined here.
- `RealAliasPairProducer`: observed variation. An entity's anchor name paired with each of its
  other recorded name variants, as carried by the canonical-stage `*-names-*.parquet` sidecars
  (GLEIF's previous-name and successor chains, Wikidata's official/short/alias/label forms).
- `NameShortNamePairProducer`: derived variation. A record's name paired with the `short_name`
  `company_cleanse` reduced it to. Cheapest of the three, and the only one whose second member
  is a deterministic transform of the first, which carries one caveat worth being deliberate
  about: if `short_name` (or anything derived from it) also becomes an input feature of the
  model being trained, this source teaches the model to recognize its own preprocessing step
  rather than real variation. Keep the representation that generates a pair distinct from the
  representation that feeds the model.

No producer computes its own split. Each takes the shared `EntitySplitter` and tags every pair
with the partition it reports for that entity, which is what makes a real-versus-synthetic
comparison possible without a separate holdout mechanism: real-alias pairs whose entity lands in
`test` were never trainable, whatever mixture of producers supplied the training pairs.
Negatives come from `negatives.py`, shared for the same reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from company_perturbation import (
    PerturbationProfile,
    SitedOperatorRegistry,
    SourceRecord,
    perturb_records,
    sited_registry,
)

from .entity_split import EntitySplitter, SplitName
from .negatives import NegativeSamplingConfig, sample_negatives
from .pairs import MATCH_LABEL, EntityRecord, NameVariant, PairRecord, PairSource

DEFAULT_ANCHOR_NAME_TYPES = ("primary", "official", "label", "short", "alias")


class PairProducer(Protocol):
    """What every producer exposes, so callers can hold a list of mixed producers."""

    @property
    def source(self) -> PairSource:
        """Which producer this is, as tagged onto every pair it emits."""

    def generate(self) -> list[PairRecord]:
        """Produce this source's positives plus their sampled negatives."""


def collect_pairs(producers: Iterable[PairProducer]) -> list[PairRecord]:
    """Run producers in order and concatenate their pairs."""
    pairs: list[PairRecord] = []
    for producer in producers:
        pairs.extend(producer.generate())
    return pairs


def _positive(
    *,
    left_name: str,
    right_name: str,
    system_uri: str,
    source: PairSource,
    splits: dict[str, SplitName],
    detail: str | None = None,
) -> PairRecord:
    return PairRecord(
        left_name=left_name,
        right_name=right_name,
        label=MATCH_LABEL,
        source=source,
        split=splits[system_uri],
        left_system_uri=system_uri,
        right_system_uri=system_uri,
        detail=detail,
    )


def _assign_splits(
    splitter: EntitySplitter, system_uris: Sequence[str]
) -> dict[str, SplitName]:
    distinct = sorted(set(system_uris))
    return dict(zip(distinct, splitter.assign_many(distinct), strict=True))


@dataclass(frozen=True)
class SyntheticPairProducer:
    """Positives from `company_perturbation`'s operators: a name and a perturbation of it.

    Attributes:
        records: Entities to perturb. `system`/`country` drive the profile's scenario
            exclusions, exactly as they do for a validation perturbation run.
        profile: The perturbation profile to apply.
        splitter: The one shared entity splitter.
        seed: The run's seed. Each record's own generator is derived from it and the
            record's id, so a producer is reproducible without a record having only one
            possible perturbation: a second producer under a different seed draws an
            independent training sample from the same entities.
        negatives: Negative-sampling policy.
        registry: Operator registry the profile's `operator_id`s resolve against.

    One record yields at most one positive: it draws a single scenario, as it would in a
    materialization run. A record whose chain landed nothing still produces a row there,
    to keep the perturbed set joinable one-to-one against its source, but a row whose
    name did not move is not a training pair -- so those are dropped here rather than
    pairing a name with itself.
    """

    records: Sequence[EntityRecord]
    profile: PerturbationProfile
    splitter: EntitySplitter
    seed: int
    negatives: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    registry: SitedOperatorRegistry = sited_registry

    @property
    def source(self) -> PairSource:
        return PairSource.SYNTHETIC

    def generate(self) -> list[PairRecord]:
        if not self.records:
            return []

        names = {record.system_uri: record.name for record in self.records}
        splits = _assign_splits(self.splitter, list(names))
        sources = [
            SourceRecord(
                local_id=record.system_uri,
                name=record.name,
                system=record.system,
                country=record.country,
            )
            for record in self.records
        ]

        positives = [
            _positive(
                left_name=names[perturbed.local_id],
                right_name=perturbed.name,
                system_uri=perturbed.local_id,
                source=self.source,
                splits=splits,
                detail=f"scenario={perturbed.scenario_id}",
            )
            for perturbed in perturb_records(
                sources, self.profile, seed=self.seed, registry=self.registry
            )
            if perturbed.changed
        ]

        return positives + sample_negatives(
            positives,
            pool=self.records,
            splitter=self.splitter,
            source=self.source,
            config=self.negatives,
        )


@dataclass(frozen=True)
class RealAliasPairProducer:
    """Positives from observed name variants of one entity.

    Attributes:
        variants: Rows of a `*-names-*.parquet` sidecar, as `NameVariant` values.
        splitter: The one shared entity splitter.
        negatives: Negative-sampling policy.
        anchor_name_types: `name_type` values in priority order; an entity's anchor is its
            highest-priority variant, falling back to the lexicographically first name when no
            variant carries a listed type. Every other distinct variant of that entity is paired
            with the anchor, so an entity with `n` distinct names yields `n - 1` positives rather
            than every combination of them.
    """

    variants: Sequence[NameVariant]
    splitter: EntitySplitter
    negatives: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    anchor_name_types: tuple[str, ...] = DEFAULT_ANCHOR_NAME_TYPES

    @property
    def source(self) -> PairSource:
        return PairSource.REAL_ALIAS

    def _anchor_rank(self, variant: NameVariant) -> tuple[int, str]:
        name_type = (variant.name_type or "").strip().lower()
        priority = (
            self.anchor_name_types.index(name_type)
            if name_type in self.anchor_name_types
            else len(self.anchor_name_types)
        )
        return priority, variant.name

    def generate(self) -> list[PairRecord]:
        grouped: dict[str, list[NameVariant]] = {}
        for variant in self.variants:
            if variant.name.strip():
                grouped.setdefault(variant.system_uri, []).append(variant)

        if not grouped:
            return []

        splits = _assign_splits(self.splitter, list(grouped))
        positives: list[PairRecord] = []
        anchors: list[EntityRecord] = []

        for system_uri in sorted(grouped):
            ordered = sorted(grouped[system_uri], key=self._anchor_rank)
            anchor = ordered[0]
            anchors.append(EntityRecord(system_uri=system_uri, name=anchor.name))

            seen = {anchor.name.casefold().strip()}
            for variant in ordered[1:]:
                key = variant.name.casefold().strip()
                if key in seen:
                    continue
                seen.add(key)
                positives.append(
                    _positive(
                        left_name=anchor.name,
                        right_name=variant.name,
                        system_uri=system_uri,
                        source=self.source,
                        splits=splits,
                        detail=f"name_type={variant.name_type or 'unknown'}",
                    )
                )

        return positives + sample_negatives(
            positives,
            pool=anchors,
            splitter=self.splitter,
            source=self.source,
            config=self.negatives,
        )


@dataclass(frozen=True)
class NameShortNamePairProducer:
    """Positives pairing a record's name with the `short_name` derived from it.

    Attributes:
        records: Entities carrying a populated `short_name`. Records without one, or whose
            `short_name` only differs from `name` by case or spacing, produce no pair: a name
            paired with itself is label noise, not a training signal.
        splitter: The one shared entity splitter.
        negatives: Negative-sampling policy.
    """

    records: Sequence[EntityRecord]
    splitter: EntitySplitter
    negatives: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)

    @property
    def source(self) -> PairSource:
        return PairSource.NAME_SHORT_NAME

    def generate(self) -> list[PairRecord]:
        usable = [
            record
            for record in self.records
            if record.name.strip()
            and record.short_name is not None
            and record.short_name.strip()
            and _normalized(record.short_name) != _normalized(record.name)
        ]
        if not usable:
            return []

        splits = _assign_splits(self.splitter, [record.system_uri for record in usable])
        positives = [
            _positive(
                left_name=record.name,
                right_name=record.short_name or "",
                system_uri=record.system_uri,
                source=self.source,
                splits=splits,
                detail="short_name",
            )
            for record in usable
        ]

        return positives + sample_negatives(
            positives,
            pool=usable,
            splitter=self.splitter,
            source=self.source,
            config=self.negatives,
        )


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())
