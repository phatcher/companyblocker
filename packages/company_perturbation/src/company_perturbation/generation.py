"""Generating one perturbed row per source record.

The unit above the chain. A record draws **one** scenario from the profile and runs
that scenario's chain, so a corpus of 5,000 records yields 5,000 perturbed rows.
That matches how real data goes wrong -- a record has one history, not one per kind
of corruption -- and keeps the perturbed set joinable one-to-one against its source.

**A record always emits.** If the drawn chain lands no changes, the row is written
with the name unchanged rather than skipped. Some records really are recorded
correctly, and skipping them would break the one-to-one guarantee. `changed` says
which case a row is, so nothing has to compare strings to find out.

**Seeding.** The run's seed is the caller's, and each record gets its own generator
derived from `(seed, local_id)`. Both halves earn their place: a record's result is
reproducible, independent of the order records were processed in and of how the
corpus was sliced, and a different seed gives a genuinely independent sample of the
same corpus.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from random import Random

from .chain import ChainResult, run_chain
from .profile_schema import (
    PerturbationProfile,
    Scenario,
    effective_exclusions,
)
from .sited_operator import AppliedChange, SitedOperatorRegistry, sited_registry


@dataclass(frozen=True)
class SourceRecord:
    """One record to perturb, as this package sees it: no dataframes, no URIs."""

    local_id: str
    name: str
    system: str
    country: str | None = None


@dataclass(frozen=True)
class PerturbedRecord:
    """The one row a source record produces."""

    local_id: str
    name: str
    original_name: str
    scenario_id: str | None
    changes: tuple[AppliedChange, ...]
    steps_that_landed_nothing: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.name != self.original_name


def _record_seed(seed: int, local_id: str) -> int:
    """A record's own generator seed, from the seed and the record's id.

    Length-prefixed so a local id containing a separator cannot collide two
    different records onto one stream.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for segment in (str(seed), local_id):
        encoded = segment.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return int.from_bytes(hasher.digest(), "big")


def applicable_scenarios(
    profile: PerturbationProfile, record: SourceRecord
) -> tuple[Scenario, ...]:
    """The scenarios this record could draw, after exclusions."""
    applicable: list[Scenario] = []
    for scenario in profile.scenarios:
        rules = effective_exclusions(scenario, profile=profile)
        if record.system in rules.exclude_systems:
            continue
        if record.country is not None and record.country in rules.exclude_countries:
            continue
        if any(
            step.operator_id.split(".", 1)[0] in rules.exclude_families
            for step in scenario.chain
        ):
            continue
        applicable.append(scenario)
    return tuple(applicable)


def select_scenario(scenarios: Sequence[Scenario], *, rng: Random) -> Scenario | None:
    """Draw one scenario, weighted. Always consumes exactly one value."""
    draw = rng.random()
    if not scenarios:
        return None

    total = sum(scenario.weight for scenario in scenarios)
    if total <= 0:
        return None

    target = draw * total
    running = 0.0
    for scenario in scenarios:
        running += scenario.weight
        if target < running:
            return scenario
    return scenarios[-1]


def perturb_record(
    record: SourceRecord,
    profile: PerturbationProfile,
    *,
    seed: int,
    registry: SitedOperatorRegistry = sited_registry,
) -> PerturbedRecord:
    """One record in, exactly one perturbed record out."""
    rng = Random(_record_seed(seed, record.local_id))  # nosec B311 - seeded for reproducibility, not security
    scenario = select_scenario(applicable_scenarios(profile, record), rng=rng)

    if scenario is None:
        return PerturbedRecord(
            local_id=record.local_id,
            name=record.name,
            original_name=record.name,
            scenario_id=None,
            changes=(),
            steps_that_landed_nothing=(),
        )

    result: ChainResult = run_chain(
        record.name,
        scenario.chain,
        rng=rng,
        registry=registry,
        country=record.country,
    )

    return PerturbedRecord(
        local_id=record.local_id,
        name=result.name,
        original_name=record.name,
        scenario_id=scenario.scenario_id,
        changes=result.changes,
        steps_that_landed_nothing=result.steps_that_landed_nothing,
    )


def perturb_records(
    records: Iterable[SourceRecord],
    profile: PerturbationProfile,
    *,
    seed: int,
    registry: SitedOperatorRegistry = sited_registry,
) -> Iterator[PerturbedRecord]:
    """One perturbed record per source record, in the order they arrive."""
    for record in records:
        yield perturb_record(record, profile, seed=seed, registry=registry)
