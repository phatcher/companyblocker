import random

import polars as pl
import pytest
from company_classify import EntitySplitter, SplitName, SplitRatios


def _entities(count: int) -> list[str]:
    return [f"gleif://LEI{index:06d}" for index in range(count)]


def test_two_differently_ordered_entity_lists_agree_on_every_entity() -> None:
    entities = _entities(500)
    shuffled = entities.copy()
    random.Random(7).shuffle(shuffled)
    splitter = EntitySplitter(seed=42)

    forward = dict(zip(entities, splitter.assign_many(entities), strict=True))
    reordered = dict(zip(shuffled, splitter.assign_many(shuffled), strict=True))

    assert forward == reordered


def test_disjoint_producer_slices_agree_with_the_whole_list() -> None:
    entities = _entities(200)
    splitter = EntitySplitter(seed=42)
    whole = dict(zip(entities, splitter.assign_many(entities), strict=True))

    first_half = entities[:80]
    second_half = entities[80:]
    partial = dict(zip(first_half, splitter.assign_many(first_half), strict=True))
    partial.update(zip(second_half, splitter.assign_many(second_half), strict=True))

    assert partial == whole


def test_scalar_assignment_matches_batch_assignment() -> None:
    entities = _entities(50)
    splitter = EntitySplitter(seed=13)

    batch = splitter.assign_many(entities)

    assert [splitter.assign(entity) for entity in entities] == batch


def test_vectorized_expression_matches_scalar_assignment() -> None:
    entities = _entities(100)
    splitter = EntitySplitter(seed=42)
    frame = pl.DataFrame({"system_uri": entities})

    column = frame.select(splitter.split_expr()).to_series().to_list()

    assert column == [splitter.assign(entity).value for entity in entities]


def test_ratios_are_respected_within_sampling_noise() -> None:
    entities = _entities(20_000)
    splitter = EntitySplitter(seed=42)

    assignments = splitter.assign_many(entities)
    fractions = {name: assignments.count(name) / len(assignments) for name in SplitName}

    assert fractions[SplitName.TRAIN] == pytest.approx(0.70, abs=0.02)
    assert fractions[SplitName.VALIDATION] == pytest.approx(0.15, abs=0.02)
    assert fractions[SplitName.TEST] == pytest.approx(0.15, abs=0.02)


def test_custom_ratios_shift_membership() -> None:
    entities = _entities(5_000)
    splitter = EntitySplitter(seed=42, ratios=SplitRatios(train=0.5, validation=0.25))

    assignments = splitter.assign_many(entities)

    assert assignments.count(SplitName.TRAIN) / len(assignments) == pytest.approx(
        0.5, abs=0.03
    )
    assert assignments.count(SplitName.TEST) / len(assignments) == pytest.approx(
        0.25, abs=0.03
    )


def test_neighbouring_seeds_produce_independent_splits() -> None:
    entities = _entities(5_000)
    base = EntitySplitter(seed=42).assign_many(entities)
    neighbour = EntitySplitter(seed=43).assign_many(entities)

    agreement = sum(
        1 for left, right in zip(base, neighbour, strict=True) if left == right
    ) / len(entities)

    # Under independence, two 70/15/15 splits agree on 0.7^2 + 0.15^2 + 0.15^2 = 0.535
    # of entities. A seeding scheme that correlated neighbouring seeds would sit far above.
    assert agreement == pytest.approx(0.535, abs=0.03)


def test_folds_partition_entities_and_are_reconstructable_per_entity() -> None:
    entities = _entities(1_000)
    splitter = EntitySplitter(seed=42)

    folds = splitter.fold_many(entities, folds=5)

    assert set(folds) == {0, 1, 2, 3, 4}
    assert [splitter.fold(entity, folds=5) for entity in entities] == folds


def test_k_fold_by_seed_reuses_the_same_split_function() -> None:
    entities = _entities(2_000)
    base = EntitySplitter(seed=42)

    folds = [base.with_seed(seed).assign_many(entities) for seed in (1, 2, 3)]

    assert all(fold != folds[0] for fold in folds[1:])
    assert base.with_seed(1).assign_many(entities) == folds[0]
    assert base.with_seed(1).ratios == base.ratios


def test_partition_groups_distinct_entities_deterministically() -> None:
    entities = _entities(300) + _entities(300)
    splitter = EntitySplitter(seed=42)

    grouped = splitter.partition(entities)

    assert set(grouped) == set(SplitName)
    assert sum(len(members) for members in grouped.values()) == 300
    for split, members in grouped.items():
        assert members == sorted(members)
        assert all(splitter.assign(member) == split for member in members)


def test_blank_entity_uri_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        EntitySplitter().assign("   ")


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="train must be in"):
        SplitRatios(train=0.0)
    with pytest.raises(ValueError, match="validation must be in"):
        SplitRatios(validation=-0.1)
    with pytest.raises(ValueError, match="must be < 1"):
        SplitRatios(train=0.9, validation=0.2)


def test_fold_count_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="folds must be at least 2"):
        EntitySplitter().fold("gleif://LEI000001", folds=1)


def test_empty_input_returns_empty_assignments() -> None:
    splitter = EntitySplitter()

    assert splitter.assign_many([]) == []
    assert splitter.fold_many([], folds=3) == []
