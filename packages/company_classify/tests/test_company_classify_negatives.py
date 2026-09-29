import pytest
from company_classify import (
    NON_MATCH_LABEL,
    EntityRecord,
    EntitySplitter,
    NegativeSamplingConfig,
    NegativeStrategy,
    PairRecord,
    PairSource,
    compose_negative_seed,
    lexical_similarity,
    sample_negatives,
    validate_pairs,
)
from company_classify.pairs import MATCH_LABEL

SPLITTER = EntitySplitter(seed=42)


def _pool(count: int = 60) -> list[EntityRecord]:
    return [
        EntityRecord(
            system_uri=f"gleif://LEI{index:06d}",
            name=f"company {index:03d} holdings limited",
            system="gleif",
        )
        for index in range(count)
    ]


def _positives(pool, count: int = 12) -> list[PairRecord]:
    return [
        PairRecord(
            left_name=record.name,
            right_name=f"{record.name} ltd",
            label=MATCH_LABEL,
            source=PairSource.SYNTHETIC,
            split=SPLITTER.assign(record.system_uri),
            left_system_uri=record.system_uri,
            right_system_uri=record.system_uri,
        )
        for record in pool[:count]
    ]


def _sample(pool, positives, **kwargs) -> list[PairRecord]:
    return sample_negatives(
        positives,
        pool=pool,
        splitter=SPLITTER,
        source=PairSource.SYNTHETIC,
        config=NegativeSamplingConfig(**kwargs),
    )


def test_negatives_pair_two_different_entities_in_one_partition() -> None:
    pool = _pool()
    positives = _positives(pool)

    negatives = _sample(pool, positives)

    assert negatives
    for negative in negatives:
        assert negative.label == NON_MATCH_LABEL
        assert negative.left_system_uri != negative.right_system_uri
        assert SPLITTER.assign(negative.left_system_uri) == negative.split
        assert SPLITTER.assign(negative.right_system_uri) == negative.split


def test_generated_pairs_satisfy_the_shared_contract() -> None:
    pool = _pool()
    positives = _positives(pool)

    validate_pairs(positives + _sample(pool, positives), splitter=SPLITTER)


def test_sampling_is_reproducible_for_the_same_inputs() -> None:
    pool = _pool()
    positives = _positives(pool)

    assert _sample(pool, positives) == _sample(pool, positives)


def test_sampling_does_not_depend_on_the_order_positives_were_generated_in() -> None:
    pool = _pool()
    positives = _positives(pool)
    reversed_positives = list(reversed(positives))

    forward = _sample(pool, positives)
    backward = _sample(pool, reversed_positives)

    assert sorted(forward, key=repr) == sorted(backward, key=repr)


def test_a_different_seed_draws_different_counterparts() -> None:
    pool = _pool()
    positives = _positives(pool)

    assert _sample(pool, positives, seed=1) != _sample(pool, positives, seed=2)


def test_negatives_per_positive_scales_the_output() -> None:
    pool = _pool()
    positives = _positives(pool)

    assert len(_sample(pool, positives, negatives_per_positive=2)) == 2 * len(positives)


def test_fractional_negatives_per_positive_lands_between_the_whole_counts() -> None:
    pool = _pool(400)
    positives = _positives(pool, count=200)

    drawn = len(_sample(pool, positives, negatives_per_positive=0.5))

    assert 0 < drawn < len(positives)


def test_zero_negatives_per_positive_produces_nothing() -> None:
    pool = _pool()
    positives = _positives(pool)

    assert _sample(pool, positives, negatives_per_positive=0) == []


def test_hard_negatives_are_lexically_closer_than_random_ones() -> None:
    pool = [
        EntityRecord(
            system_uri=f"gleif://LEI{index:06d}",
            name=name,
            system="gleif",
        )
        for index, name in enumerate(
            [
                "northwind trading limited",
                "northwind trading holdings limited",
                "northwind traders limited",
                "zephyr aviation plc",
                "kestrel foods gmbh",
                "aurora mining corporation",
                "borealis shipping as",
                "cascade paper mills inc",
            ]
        )
    ]
    anchor = pool[0]
    positive = PairRecord(
        left_name=anchor.name,
        right_name="northwind trading ltd",
        label=MATCH_LABEL,
        source=PairSource.REAL_ALIAS,
        split=SPLITTER.assign(anchor.system_uri),
        left_system_uri=anchor.system_uri,
        right_system_uri=anchor.system_uri,
    )
    same_partition = [
        record
        for record in pool
        if SPLITTER.assign(record.system_uri) == positive.split
    ]

    hard = sample_negatives(
        [positive],
        pool=same_partition,
        splitter=SPLITTER,
        source=PairSource.REAL_ALIAS,
        config=NegativeSamplingConfig(strategy=NegativeStrategy.HARD),
    )

    assert len(hard) == 1
    best_available = max(
        lexical_similarity(anchor.name, record.name)
        for record in same_partition
        if record.system_uri != anchor.system_uri
    )
    assert lexical_similarity(anchor.name, hard[0].right_name) == pytest.approx(
        best_available
    )
    assert hard[0].detail == "negative=hard"


def test_a_partition_with_one_entity_yields_no_negatives() -> None:
    pool = _pool()
    lonely = next(record for record in pool)
    positives = _positives([lonely], count=1)

    assert _sample([lonely], positives) == []


def test_candidate_with_the_anchor_name_is_never_chosen() -> None:
    anchor = EntityRecord(system_uri="gleif://LEI000001", name="acme limited")
    twin_uri = next(
        uri
        for uri in (f"gleif://LEI{index:06d}" for index in range(2, 500))
        if SPLITTER.assign(uri) == SPLITTER.assign(anchor.system_uri)
    )
    pool = [anchor, EntityRecord(system_uri=twin_uri, name="ACME Limited")]
    positives = _positives([anchor], count=1)

    assert _sample(pool, positives) == []


def test_similarity_is_symmetric_and_bounded() -> None:
    assert lexical_similarity("acme limited", "acme limited") == 1.0
    assert lexical_similarity("acme", "zzzz") == 0.0
    assert lexical_similarity("acme limited", "acme ltd") == lexical_similarity(
        "acme ltd", "acme limited"
    )


def test_seed_composition_separates_segment_boundaries() -> None:
    assert compose_negative_seed("ab", "c") != compose_negative_seed("a", "bc")
    assert compose_negative_seed("ab", "c") == compose_negative_seed("ab", "c")


def test_invalid_sampling_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        NegativeSamplingConfig(negatives_per_positive=-1)
    with pytest.raises(ValueError, match="hard_pool_size must be at least 1"):
        NegativeSamplingConfig(hard_pool_size=0)
