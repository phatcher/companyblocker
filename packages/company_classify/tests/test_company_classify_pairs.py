import pytest
from company_classify import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    EntitySplitter,
    NameVariant,
    PairContractError,
    PairRecord,
    PairSource,
    SplitName,
    find_ambiguous_pairs,
    find_contract_violations,
    find_split_leakage,
    name_variants_from_rows,
    validate_pairs,
)

SPLITTER = EntitySplitter(seed=42)

LEFT = "gleif://LEI000001"
RIGHT = "gleif://LEI000002"


def _pair(
    *,
    left_name: str = "acme systems limited",
    right_name: str = "acme systems ltd",
    label: int = MATCH_LABEL,
    left_system_uri: str = LEFT,
    right_system_uri: str | None = None,
    split: SplitName | None = None,
    source: PairSource = PairSource.REAL_ALIAS,
) -> PairRecord:
    right_uri = right_system_uri if right_system_uri is not None else left_system_uri
    return PairRecord(
        left_name=left_name,
        right_name=right_name,
        label=label,
        source=source,
        split=split if split is not None else SPLITTER.assign(left_system_uri),
        left_system_uri=left_system_uri,
        right_system_uri=right_uri,
    )


def test_a_well_formed_pair_set_passes_every_check() -> None:
    negative_split = SPLITTER.assign(LEFT)
    other = next(
        uri
        for uri in (f"gleif://LEI{index:06d}" for index in range(100, 400))
        if SPLITTER.assign(uri) == negative_split
    )
    pairs = [
        _pair(),
        _pair(
            right_name="other holdings plc",
            label=NON_MATCH_LABEL,
            right_system_uri=other,
        ),
    ]

    validate_pairs(pairs, splitter=SPLITTER)


def test_match_pair_spanning_two_entities_is_a_violation() -> None:
    pair = _pair(right_system_uri=RIGHT, split=SPLITTER.assign(LEFT))

    violations = find_contract_violations([pair], splitter=SPLITTER)

    assert any("match pair spans two entities" in violation for violation in violations)


def test_non_match_pair_on_one_entity_is_a_violation() -> None:
    pair = _pair(label=NON_MATCH_LABEL)

    violations = find_contract_violations([pair], splitter=SPLITTER)

    assert any("names one entity twice" in violation for violation in violations)


def test_producer_computed_split_disagreeing_with_the_shared_split_is_a_violation() -> (
    None
):
    wrong = next(name for name in SplitName if name != SPLITTER.assign(LEFT))
    pair = _pair(split=wrong)

    violations = find_contract_violations([pair], splitter=SPLITTER)

    assert any("disagrees with the shared splitter" in v for v in violations)
    assert len(violations) == 2  # one per side of the pair


def test_non_binary_label_and_blank_members_are_violations() -> None:
    pairs = [
        _pair(label=2),
        _pair(left_name="  "),
        _pair(left_system_uri=LEFT, right_system_uri=""),
    ]

    violations = find_contract_violations(pairs, splitter=SPLITTER)

    assert any("is not 0 or 1" in violation for violation in violations)
    assert any("non-empty names" in violation for violation in violations)
    assert any("must carry a system_uri" in violation for violation in violations)


def test_split_leakage_is_reported_per_entity() -> None:
    pairs = [
        _pair(split=SplitName.TRAIN),
        _pair(right_name="acme systems", split=SplitName.TEST),
    ]

    leakage = find_split_leakage(pairs)

    assert leakage == {LEFT: {SplitName.TRAIN, SplitName.TEST}}


def test_clean_pairs_report_no_leakage() -> None:
    assert find_split_leakage([_pair()]) == {}


def test_conflicting_labels_for_one_name_pair_are_ambiguous() -> None:
    pairs = [
        _pair(),
        _pair(
            label=NON_MATCH_LABEL,
            right_system_uri=RIGHT,
            split=SPLITTER.assign(LEFT),
        ),
    ]

    ambiguities = find_ambiguous_pairs(pairs)

    assert any("labelled both match and non-match" in item for item in ambiguities)


def test_a_pair_of_identical_names_is_ambiguous() -> None:
    pair = _pair(right_name="acme systems limited")

    ambiguities = find_ambiguous_pairs([pair])

    assert any("both members are the same text" in item for item in ambiguities)


def test_validate_pairs_raises_listing_every_problem() -> None:
    pairs = [_pair(label=7), _pair(right_name="acme systems limited")]

    with pytest.raises(PairContractError) as excinfo:
        validate_pairs(pairs, splitter=SPLITTER)

    message = str(excinfo.value)
    assert "is not 0 or 1" in message
    assert "both members are the same text" in message


def test_empty_pair_sets_are_trivially_valid() -> None:
    assert find_contract_violations([], splitter=SPLITTER) == []
    validate_pairs([], splitter=SPLITTER)


def test_name_variant_rows_are_adapted_from_sidecar_columns() -> None:
    rows: list[dict[str, object]] = [
        {
            "system_uri": "wikidata://Q1201",
            "id": "Q1201",
            "name": "Saarland",
            "source_type": "LABEL_EN",
            "language_code": "en",
            "derivation_note": "wikidata labels.en",
            "name_type": "label",
        },
        {"system_uri": "wikidata://Q1201", "name": "Saarland Land", "name_type": None},
    ]

    variants = name_variants_from_rows(rows)

    assert variants == [
        NameVariant(system_uri="wikidata://Q1201", name="Saarland", name_type="label"),
        NameVariant(
            system_uri="wikidata://Q1201", name="Saarland Land", name_type=None
        ),
    ]
