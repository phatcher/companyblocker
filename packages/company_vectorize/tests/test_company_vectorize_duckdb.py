import pytest
from company_vectorize.duckdb_sql import (
    create_exact_knn_query_sql,
    create_table_sql_multi,
    create_upsert_sql_multi,
)


def test_create_table_sql_multi_builds_text_and_multiple_vector_columns():
    sql = create_table_sql_multi(
        table="company_vectors",
        vector_dimensions={"name_vec": 384, "short_vec": 128},
        id_col="entity_id",
        text_cols=("name_cleansed", "short_name"),
    )

    assert "entity_id VARCHAR PRIMARY KEY" in sql
    assert "name_cleansed VARCHAR" in sql
    assert "short_name VARCHAR" in sql
    assert "name_vec DOUBLE[]" in sql
    assert "short_vec DOUBLE[]" in sql
    assert "metadata JSON" in sql


def test_create_table_sql_multi_validates_dimensions():
    with pytest.raises(ValueError, match="at least one vector column"):
        create_table_sql_multi(table="company_vectors", vector_dimensions={})
    with pytest.raises(ValueError, match="must be greater than zero"):
        create_table_sql_multi(
            table="company_vectors", vector_dimensions={"name_vec": 0}
        )


def test_create_upsert_sql_multi_includes_all_columns_and_conflict_clause():
    sql = create_upsert_sql_multi(
        table="company_vectors",
        id_col="entity_id",
        text_cols=("name_cleansed", "short_name"),
        vector_cols=("name_vec", "short_vec"),
    )

    assert "INSERT INTO company_vectors" in sql
    assert "ON CONFLICT (entity_id) DO UPDATE" in sql
    assert "name_cleansed = EXCLUDED.name_cleansed" in sql
    assert "short_vec = EXCLUDED.short_vec" in sql
    assert "metadata = EXCLUDED.metadata" in sql


def test_create_upsert_sql_multi_validates_vector_cols():
    with pytest.raises(ValueError, match="at least one vector column"):
        create_upsert_sql_multi(
            table="company_vectors",
            id_col="entity_id",
            text_cols=("name",),
            vector_cols=(),
        )


def test_create_exact_knn_query_sql_supports_multiple_metrics():
    cosine_sql = create_exact_knn_query_sql(
        table="company_vectors",
        vector_col="short_vec",
        distance="cosine",
        limit=7,
    )
    assert "list_cosine_distance(short_vec, ?::DOUBLE[])" in cosine_sql
    assert "LIMIT 7" in cosine_sql

    l2_sql = create_exact_knn_query_sql(
        table="company_vectors",
        vector_col="short_vec",
        distance="l2",
    )
    assert "list_distance(short_vec, ?::DOUBLE[])" in l2_sql

    dot_sql = create_exact_knn_query_sql(
        table="company_vectors",
        vector_col="short_vec",
        distance="dot",
    )
    assert "-list_dot_product(short_vec, ?::DOUBLE[])" in dot_sql


def test_create_exact_knn_query_sql_validates_limit_and_metric():
    with pytest.raises(ValueError, match="limit"):
        create_exact_knn_query_sql(
            table="company_vectors", vector_col="name_vec", limit=0
        )
    with pytest.raises(ValueError, match="Unsupported DuckDB distance metric"):
        create_exact_knn_query_sql(
            table="company_vectors",
            vector_col="name_vec",
            distance="invalid",  # type: ignore[arg-type]
        )
