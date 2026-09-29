import pytest
from company_vectorize.pgvector import (
    create_ann_query_sql,
    create_extension_sql,
    create_hnsw_index_sql,
    create_table_sql,
    create_upsert_sql,
    vector_literal,
)


def test_create_extension_sql():
    assert create_extension_sql() == "CREATE EXTENSION IF NOT EXISTS vector;"


def test_create_table_sql_includes_vector_dimensions_and_metadata():
    sql = create_table_sql(table="company_vectors", dimensions=384)

    assert "vector(384)" in sql
    assert "metadata jsonb" in sql


def test_create_table_sql_validates_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        create_table_sql(table="company_vectors", dimensions=0)


def test_create_hnsw_index_sql_for_cosine():
    sql = create_hnsw_index_sql(
        table="company_vectors", distance="cosine", m=24, ef_construction=128
    )

    assert "USING hnsw" in sql
    assert "vector_cosine_ops" in sql
    assert "m = 24" in sql
    assert "ef_construction = 128" in sql


def test_create_hnsw_index_sql_validates_parameters():
    with pytest.raises(ValueError, match="m"):
        create_hnsw_index_sql(table="company_vectors", m=0)
    with pytest.raises(ValueError, match="ef_construction"):
        create_hnsw_index_sql(table="company_vectors", ef_construction=0)


def test_create_upsert_sql_includes_casts_and_conflict_clause():
    sql = create_upsert_sql(table="company_vectors")

    assert "%s::vector" in sql
    assert "ON CONFLICT" in sql


def test_create_ann_query_sql_uses_distance_operator_and_limit():
    sql = create_ann_query_sql(table="company_vectors", distance="l2", limit=5)

    assert "ORDER BY name_vec <-> %s::vector" in sql
    assert "LIMIT 5" in sql


def test_create_ann_query_sql_validates_limit():
    with pytest.raises(ValueError, match="limit"):
        create_ann_query_sql(table="company_vectors", limit=0)


def test_vector_literal_formats_pgvector_array():
    assert vector_literal([0.1, -1.25, 2.0]) == "[0.1,-1.25,2]"
