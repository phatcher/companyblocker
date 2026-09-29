import pytest
from company_classify import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    EntityRecord,
    EntitySplitter,
    NameShortNamePairProducer,
    NameVariant,
    NegativeSamplingConfig,
    PairProducer,
    PairSource,
    RealAliasPairProducer,
    SplitName,
    SyntheticPairProducer,
    collect_pairs,
    validate_pairs,
)
from company_perturbation import (
    ChainStep,
    PerturbationProfile,
    Scenario,
)

SPLITTER = EntitySplitter(seed=42)
NO_NEGATIVES = NegativeSamplingConfig(negatives_per_positive=0)
SEED = 11


def _profile():
    return PerturbationProfile(
        profile_id="pair-generation-v1",
        scenarios=(
            Scenario(
                scenario_id="light-typo",
                chain=(ChainStep("typo.keyboard_substitution", 1.0, 2),),
            ),
            Scenario(
                scenario_id="suffix-variant",
                chain=(ChainStep("legal_suffix.variant_substitution", 1.0, 1),),
            ),
        ),
    )


def _records(count: int = 40) -> list[EntityRecord]:
    return [
        EntityRecord(
            system_uri=f"gb://{index:06d}",
            name=f"northwind trading {index:03d} limited",
            system="gb",
            country="gb",
            short_name=f"northwind {index:03d}",
        )
        for index in range(count)
    ]


def _variants() -> list[NameVariant]:
    return [
        NameVariant("gleif://LEI000001", "ACME SYSTEMS LIMITED", "primary"),
        NameVariant("gleif://LEI000001", "ACME SYSTEMS LTD", "previous"),
        NameVariant("gleif://LEI000001", "ACME SYSTEMS", "trading"),
        NameVariant("gleif://LEI000002", "BOREALIS SHIPPING AS", "primary"),
        NameVariant("gleif://LEI000002", "BOREALIS SHIPPING", "trading"),
        NameVariant(
            "wikidata://Q4614", "University of Southern California", "official"
        ),
        NameVariant("wikidata://Q4614", "USC", "short"),
        NameVariant("wikidata://Q4614", "USC Trojans", "alias"),
    ]


def test_synthetic_producer_pairs_each_name_with_its_perturbation() -> None:
    records = _records()
    producer = SyntheticPairProducer(
        records=records,
        profile=_profile(),
        seed=SEED,
        splitter=SPLITTER,
        negatives=NO_NEGATIVES,
    )

    pairs = producer.generate()

    assert pairs
    names = {record.system_uri: record.name for record in records}
    for pair in pairs:
        assert pair.source == PairSource.SYNTHETIC
        assert pair.label == MATCH_LABEL
        assert pair.left_system_uri == pair.right_system_uri
        assert pair.left_name == names[pair.left_system_uri]
        assert pair.right_name != pair.left_name
        assert pair.detail is not None and pair.detail.startswith("scenario=")


def test_synthetic_producer_is_reproducible() -> None:
    producer = SyntheticPairProducer(
        records=_records(),
        profile=_profile(),
        seed=SEED,
        splitter=SPLITTER,
        negatives=NO_NEGATIVES,
    )

    assert producer.generate() == producer.generate()


def test_synthetic_producer_draws_an_independent_sample_under_a_different_seed() -> (
    None
):
    """The same entities can supply more than one training sample."""

    def _pairs(seed: int) -> list[str]:
        return [
            pair.right_name
            for pair in SyntheticPairProducer(
                records=_records(),
                profile=_profile(),
                seed=seed,
                splitter=SPLITTER,
                negatives=NO_NEGATIVES,
            ).generate()
        ]

    assert _pairs(SEED) != _pairs(SEED + 1)


def test_synthetic_producer_never_pairs_a_name_with_itself() -> None:
    """A chain that landed nothing emits a row, but an unmoved name is not a pair."""
    records = [
        EntityRecord(
            system_uri=f"gb://{index:06d}",
            name=f"northwind trading {index:03d}",
            system="gb",
            country="gb",
        )
        for index in range(20)
    ]
    profile = PerturbationProfile(
        profile_id="suffix-only",
        scenarios=(
            Scenario(
                scenario_id="drop", chain=(ChainStep("legal_suffix.drop", 1.0, 1),)
            ),
        ),
    )

    pairs = SyntheticPairProducer(
        records=records,
        profile=profile,
        seed=SEED,
        splitter=SPLITTER,
        negatives=NO_NEGATIVES,
    ).generate()

    assert pairs == []


def test_real_alias_producer_pairs_variants_against_the_entity_anchor() -> None:
    producer = RealAliasPairProducer(
        variants=_variants(), splitter=SPLITTER, negatives=NO_NEGATIVES
    )

    pairs = producer.generate()

    assert len(pairs) == 5
    by_entity: dict[str, list] = {}
    for pair in pairs:
        assert pair.source == PairSource.REAL_ALIAS
        assert pair.label == MATCH_LABEL
        by_entity.setdefault(pair.left_system_uri, []).append(pair)

    assert {pair.left_name for pair in by_entity["gleif://LEI000001"]} == {
        "ACME SYSTEMS LIMITED"
    }
    assert {pair.right_name for pair in by_entity["wikidata://Q4614"]} == {
        "USC",
        "USC Trojans",
    }
    assert all(
        pair.left_name == "University of Southern California"
        for pair in by_entity["wikidata://Q4614"]
    )


def test_real_alias_anchor_priority_is_configurable() -> None:
    producer = RealAliasPairProducer(
        variants=_variants(),
        splitter=SPLITTER,
        negatives=NO_NEGATIVES,
        anchor_name_types=("short", "trading"),
    )

    pairs = producer.generate()

    anchors = {pair.left_system_uri: pair.left_name for pair in pairs}
    assert anchors["wikidata://Q4614"] == "USC"
    assert anchors["gleif://LEI000001"] == "ACME SYSTEMS"


def test_real_alias_producer_drops_duplicate_variant_text() -> None:
    variants = [
        NameVariant("gleif://LEI000001", "ACME SYSTEMS LIMITED", "primary"),
        NameVariant("gleif://LEI000001", "acme systems limited", "previous"),
        NameVariant("gleif://LEI000001", "ACME SYSTEMS", "trading"),
    ]

    pairs = RealAliasPairProducer(
        variants=variants, splitter=SPLITTER, negatives=NO_NEGATIVES
    ).generate()

    assert [pair.right_name for pair in pairs] == ["ACME SYSTEMS"]


def test_name_short_name_producer_pairs_a_record_with_its_own_short_name() -> None:
    records = _records()

    pairs = NameShortNamePairProducer(
        records=records, splitter=SPLITTER, negatives=NO_NEGATIVES
    ).generate()

    assert len(pairs) == len(records)
    for pair, record in zip(pairs, records, strict=True):
        assert pair.source == PairSource.NAME_SHORT_NAME
        assert pair.left_name == record.name
        assert pair.right_name == record.short_name
        assert pair.detail == "short_name"


def test_name_short_name_producer_skips_missing_or_equivalent_short_names() -> None:
    records = [
        EntityRecord(system_uri="gb://000001", name="acme limited"),
        EntityRecord(
            system_uri="gb://000002", name="acme limited", short_name="  ACME Limited "
        ),
        EntityRecord(system_uri="gb://000003", name="acme limited", short_name="acme"),
    ]

    pairs = NameShortNamePairProducer(
        records=records, splitter=SPLITTER, negatives=NO_NEGATIVES
    ).generate()

    assert [pair.left_system_uri for pair in pairs] == ["gb://000003"]


def test_every_producer_tags_the_one_shared_split_never_its_own() -> None:
    records = _records()
    variants = _variants()
    producers: list[PairProducer] = [
        SyntheticPairProducer(
            records=records, profile=_profile(), seed=SEED, splitter=SPLITTER
        ),
        RealAliasPairProducer(variants=variants, splitter=SPLITTER),
        NameShortNamePairProducer(records=records, splitter=SPLITTER),
    ]

    pairs = collect_pairs(producers)

    assert {pair.source for pair in pairs} == set(PairSource)
    validate_pairs(pairs, splitter=SPLITTER)
    for pair in pairs:
        assert pair.split == SPLITTER.assign(pair.left_system_uri)


def test_a_producer_using_a_different_seed_is_caught_by_the_contract_check() -> None:
    from company_classify import PairContractError

    pairs = NameShortNamePairProducer(
        records=_records(), splitter=SPLITTER.with_seed(99), negatives=NO_NEGATIVES
    ).generate()

    with pytest.raises(PairContractError, match="disagrees with the shared splitter"):
        validate_pairs(pairs, splitter=SPLITTER)


def test_producers_emit_negatives_alongside_positives_by_default() -> None:
    records = _records()

    pairs = NameShortNamePairProducer(records=records, splitter=SPLITTER).generate()

    negatives = [pair for pair in pairs if pair.label == NON_MATCH_LABEL]
    assert negatives
    for negative in negatives:
        assert negative.source == PairSource.NAME_SHORT_NAME
        assert negative.left_system_uri != negative.right_system_uri
        assert negative.split == SPLITTER.assign(negative.right_system_uri)


def test_real_alias_test_partition_pairs_survive_a_train_only_mixture() -> None:
    """Real-alias `test` pairs are never trainable, whichever producers supply training data."""

    records = _records(200)
    variants = [
        NameVariant(record.system_uri, record.name, "primary") for record in records
    ] + [
        NameVariant(record.system_uri, f"{record.name} holdings", "previous")
        for record in records
    ]

    trained_on = [
        pair
        for pair in collect_pairs(
            [
                SyntheticPairProducer(
                    records=records, profile=_profile(), seed=SEED, splitter=SPLITTER
                ),
                NameShortNamePairProducer(records=records, splitter=SPLITTER),
            ]
        )
        if pair.split == SplitName.TRAIN
    ]
    held_out = [
        pair
        for pair in RealAliasPairProducer(
            variants=variants, splitter=SPLITTER
        ).generate()
        if pair.split == SplitName.TEST
    ]

    assert trained_on and held_out
    trained_entities = {pair.left_system_uri for pair in trained_on} | {
        pair.right_system_uri for pair in trained_on
    }
    held_out_entities = {pair.left_system_uri for pair in held_out} | {
        pair.right_system_uri for pair in held_out
    }
    assert trained_entities.isdisjoint(held_out_entities)


def test_producers_with_no_usable_input_produce_nothing() -> None:
    assert (
        SyntheticPairProducer(
            records=[], profile=_profile(), seed=SEED, splitter=SPLITTER
        ).generate()
        == []
    )
    assert RealAliasPairProducer(variants=[], splitter=SPLITTER).generate() == []
    assert NameShortNamePairProducer(records=[], splitter=SPLITTER).generate() == []
