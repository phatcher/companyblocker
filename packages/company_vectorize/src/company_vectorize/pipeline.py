from __future__ import annotations

from .base import EmbeddedRow, MultiEmbeddedRow, Vectorizer, l2_normalize


def vectorize_records(
    records: list[dict[str, object]],
    *,
    vectorizer: Vectorizer,
    id_col: str = "entity_id",
    text_col: str = "name",
    normalize_vectors: bool = True,
    skip_empty: bool = True,
) -> list[EmbeddedRow]:
    prepared_ids: list[str] = []
    prepared_texts: list[str] = []

    for record in records:
        entity_id = str(record.get(id_col, "")).strip()
        text = str(record.get(text_col, "")).strip()
        if not entity_id:
            continue
        if not text and skip_empty:
            continue
        prepared_ids.append(entity_id)
        prepared_texts.append(text)

    vectors = vectorizer.embed_texts(prepared_texts)
    if len(vectors) != len(prepared_ids):
        raise ValueError(
            "Vectorizer returned a vector count that does not match input rows."
        )

    rows: list[EmbeddedRow] = []
    for entity_id, text, vector in zip(
        prepared_ids, prepared_texts, vectors, strict=True
    ):
        if normalize_vectors:
            vector = l2_normalize(vector)
        rows.append(EmbeddedRow(entity_id=entity_id, text=text, vector=vector))

    return rows


def vectorize_records_multi(
    records: list[dict[str, object]],
    *,
    vectorizers_by_target: dict[str, Vectorizer],
    text_cols_by_target: dict[str, str],
    id_col: str = "entity_id",
    normalize_vectors: bool = True,
    skip_empty: bool = True,
) -> list[MultiEmbeddedRow]:
    if not vectorizers_by_target:
        raise ValueError("vectorizers_by_target must contain at least one target.")
    if set(vectorizers_by_target.keys()) != set(text_cols_by_target.keys()):
        raise ValueError(
            "vectorizers_by_target keys must match text_cols_by_target keys."
        )

    target_buffers: dict[str, list[str]] = {
        target: [] for target in vectorizers_by_target
    }
    prepared_ids: list[str] = []
    prepared_texts_by_target: list[dict[str, str]] = []

    for record in records:
        entity_id = str(record.get(id_col, "")).strip()
        if not entity_id:
            continue

        row_texts: dict[str, str] = {}
        has_any_text = False
        for target, text_col in text_cols_by_target.items():
            text = str(record.get(text_col, "")).strip()
            row_texts[target] = text
            if text:
                has_any_text = True
            target_buffers[target].append(text)

        if skip_empty and not has_any_text:
            # Remove the buffered values to keep vectors aligned with accepted ids.
            for buf in target_buffers.values():
                buf.pop()
            continue

        prepared_ids.append(entity_id)
        prepared_texts_by_target.append(row_texts)

    vectors_by_target: dict[str, list[list[float]]] = {}
    for target, vectorizer in vectorizers_by_target.items():
        vectors = vectorizer.embed_texts(target_buffers[target])
        if len(vectors) != len(prepared_ids):
            raise ValueError(
                f"Vectorizer '{target}' returned a vector count that does not match input rows."
            )
        if normalize_vectors:
            vectors = [l2_normalize(vector) for vector in vectors]
        vectors_by_target[target] = vectors

    rows: list[MultiEmbeddedRow] = []
    for index, entity_id in enumerate(prepared_ids):
        row_vectors = {
            target: vectors_by_target[target][index] for target in vectorizers_by_target
        }
        rows.append(
            MultiEmbeddedRow(
                entity_id=entity_id,
                texts=prepared_texts_by_target[index],
                vectors=row_vectors,
            )
        )

    return rows
