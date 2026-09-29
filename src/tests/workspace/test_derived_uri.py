"""Direct tests for `workspace.derived_uri`.

What is pinned is that each form of derived URI round-trips through the one
composer and parser, decomposes to the source URI it was derived from with no
lookup, and less its row identity is its dataset's URI.
"""

from __future__ import annotations

import pytest

from workspace.derived_uri import (
    DerivedUri,
    Perturbation,
    compose_uri,
    name_hash,
    parse_uri,
)
from workspace.reference import RESERVED_WORDS, InvalidReferenceError

HASH = "0123456789abcdef"
EN_LITE = Perturbation(profile="en-lite", version="v1", seed=42)


def test_an_entity_uri_is_the_underived_case() -> None:
    entity = parse_uri("gb://123")

    assert entity == DerivedUri(system="gb", entity_id="123")
    assert entity.uri == "gb://123"
    assert entity.source is None
    assert not entity.derived


def test_a_name_variant_decomposes_to_its_entity() -> None:
    name = DerivedUri(system="gb", names=True, entity_id="123", name_hash=HASH)

    assert name.uri == f"name://gb/123/{HASH}"
    assert parse_uri(name.uri) == name
    assert name.source is not None
    assert name.source.uri == "gb://123"


def test_a_perturbed_entity_decomposes_to_its_entity_and_its_dataset() -> None:
    row = DerivedUri(system="ie", perturbation=EN_LITE, entity_id="123")

    assert row.uri == "perturbed://ie/en-lite/v1/42/123"
    assert parse_uri(row.uri) == row
    assert row.source is not None
    assert row.source.uri == "ie://123"
    assert row.dataset.uri == "perturbed://ie/en-lite/v1/42"
    assert parse_uri(row.dataset.uri) == row.dataset


def test_a_perturbed_name_variant_decomposes_to_its_name_variant_and_its_dataset() -> (
    None
):
    row = DerivedUri(
        system="gb", names=True, perturbation=EN_LITE, entity_id="123", name_hash=HASH
    )

    assert row.uri == f"perturbed://gb/names/en-lite/v1/42/123/{HASH}"
    assert parse_uri(row.uri) == row
    assert row.source is not None
    assert row.source.uri == f"name://gb/123/{HASH}"
    assert row.source.source is not None
    assert row.source.source.uri == "gb://123"
    assert row.dataset.uri == "perturbed://gb/names/en-lite/v1/42"
    assert parse_uri(row.dataset.uri) == row.dataset


def test_an_entity_id_holding_a_slash_is_one_segment_and_reads_back() -> None:
    name = DerivedUri(system="gb", names=True, entity_id="ab/c%2", name_hash=HASH)

    assert name.uri == f"name://gb/ab%2Fc%252/{HASH}"
    assert parse_uri(name.uri) == name
    # The entity's own URI is not re-spelled.
    assert name.source is not None
    assert name.source.uri == "gb://ab/c%2"
    assert parse_uri("gb://ab/c%2").entity_id == "ab/c%2"


def test_a_negative_seed_round_trips() -> None:
    dataset = DerivedUri(
        system="ie", perturbation=Perturbation(profile="en-lite", version="v2", seed=-7)
    )

    assert parse_uri(dataset.uri) == dataset


@pytest.mark.parametrize("word", sorted(RESERVED_WORDS))
def test_a_reserved_word_names_no_system_or_profile(word: str) -> None:
    with pytest.raises(InvalidReferenceError):
        DerivedUri(system=word, entity_id="1")
    with pytest.raises(InvalidReferenceError):
        Perturbation(profile=word, version="v1", seed=1)


@pytest.mark.parametrize("scheme", ["name", "blocking", "perturbation"])
def test_a_scheme_is_never_read_as_a_system(scheme: str) -> None:
    with pytest.raises(InvalidReferenceError):
        DerivedUri(system=scheme, entity_id="1")


@pytest.mark.parametrize(
    "uri",
    [
        "gb",
        "gb://",
        "name://gb/123",
        f"name://gb/123/{HASH}/extra",
        "name://gb/123/not-a-hash",
        "perturbed://ie/en-lite/v1",
        "perturbed://ie/en-lite/v1/x",
        f"perturbed://ie/en-lite/v1/42/123/{HASH}",
        "perturbed://gb/names/en-lite/v1/42/123",
        "perturbed://ie//v1/42",
    ],
)
def test_a_malformed_uri_is_refused(uri: str) -> None:
    with pytest.raises(InvalidReferenceError):
        parse_uri(uri)


def test_a_stored_dataset_has_no_derived_uri() -> None:
    with pytest.raises(InvalidReferenceError):
        compose_uri(DerivedUri(system="gb", names=True))


def test_a_name_hash_covers_the_type_and_value_and_not_the_entity() -> None:
    assert name_hash(name_type="alias", value="Acme") == name_hash(
        name_type="alias", value="Acme"
    )
    assert name_hash(name_type="alias", value="Acme") != name_hash(
        name_type="legal", value="Acme"
    )
    # Length-prefixed, so moving a character across the field boundary changes it.
    assert name_hash(name_type="ab", value="c") != name_hash(name_type="a", value="bc")


def test_a_name_variant_and_a_perturbed_row_are_composed_from_the_row_they_came_from() -> (
    None
):
    from workspace.derived_uri import name_variant_uri, perturbed_uri, source_uri_of

    name = name_variant_uri(source_uri="gb://123", name_type="alias", value="Acme")
    perturbed_entity = perturbed_uri(source_uri="gb://123", perturbation=EN_LITE)
    perturbed_name = perturbed_uri(source_uri=name, perturbation=EN_LITE)

    assert name == f"name://gb/123/{name_hash(name_type='alias', value='Acme')}"
    assert perturbed_entity == "perturbed://gb/en-lite/v1/42/123"
    assert source_uri_of(perturbed_name) == name
    assert source_uri_of(name) == "gb://123"
    assert source_uri_of("gb://123") is None
    # Two variants of one entity, and one variant of two entities, all differ.
    assert name != name_variant_uri(
        source_uri="gb://123", name_type="alias", value="Acme Ltd"
    )
    assert name != name_variant_uri(
        source_uri="gb://124", name_type="alias", value="Acme"
    )
    with pytest.raises(InvalidReferenceError):
        name_variant_uri(source_uri=name, name_type="alias", value="Acme")
    with pytest.raises(InvalidReferenceError):
        perturbed_uri(source_uri=perturbed_entity, perturbation=EN_LITE)
