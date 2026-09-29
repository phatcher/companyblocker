import pytest
from company_vectorize.backends import HashVectorizer
from company_vectorize.pipeline import vectorize_records, vectorize_records_multi


def test_vectorize_records_skips_empty_and_missing_ids():
    records = [
        {"entity_id": "1", "name": "Acme Ltd"},
        {"entity_id": "", "name": "Skip Me"},
        {"entity_id": "3", "name": ""},
    ]

    rows = vectorize_records(
        records, vectorizer=HashVectorizer(dimension=8), skip_empty=True
    )

    assert len(rows) == 1
    assert rows[0].entity_id == "1"
    assert rows[0].text == "Acme Ltd"
    assert len(rows[0].vector) == 8


def test_vectorize_records_detects_mismatched_vector_counts(monkeypatch):
    class BrokenVectorizer:
        @property
        def dimension(self):
            return 4

        def embed_texts(self, texts):
            return []

    with pytest.raises(ValueError, match="vector count"):
        vectorize_records(
            [{"entity_id": "1", "name": "Acme Ltd"}],
            vectorizer=BrokenVectorizer(),
        )


def test_vectorize_records_multi_returns_independent_vectors_per_target():
    records = [
        {
            "entity_id": "1",
            "name_cleansed": "Acme Holdings Ltd",
            "short_name": "Acme",
        }
    ]

    rows = vectorize_records_multi(
        records,
        vectorizers_by_target={
            "cleanse_vec": HashVectorizer(dimension=8),
            "short_vec": HashVectorizer(dimension=8),
        },
        text_cols_by_target={
            "cleanse_vec": "name_cleansed",
            "short_vec": "short_name",
        },
    )

    assert len(rows) == 1
    assert rows[0].entity_id == "1"
    assert rows[0].texts["cleanse_vec"] == "Acme Holdings Ltd"
    assert rows[0].texts["short_vec"] == "Acme"
    assert len(rows[0].vectors["cleanse_vec"]) == 8
    assert len(rows[0].vectors["short_vec"]) == 8
    assert rows[0].vectors["cleanse_vec"] != rows[0].vectors["short_vec"]


def test_vectorize_records_multi_validates_target_mapping_and_counts():
    with pytest.raises(ValueError, match="at least one target"):
        vectorize_records_multi(
            [],
            vectorizers_by_target={},
            text_cols_by_target={},
        )

    with pytest.raises(ValueError, match="keys must match"):
        vectorize_records_multi(
            [{"entity_id": "1", "name": "Acme"}],
            vectorizers_by_target={"a": HashVectorizer(dimension=4)},
            text_cols_by_target={"b": "name"},
        )

    class BrokenVectorizer:
        @property
        def dimension(self):
            return 4

        def embed_texts(self, texts):
            return []

    with pytest.raises(ValueError, match="does not match input rows"):
        vectorize_records_multi(
            [{"entity_id": "1", "name": "Acme"}],
            vectorizers_by_target={"name_vec": BrokenVectorizer()},
            text_cols_by_target={"name_vec": "name"},
        )
