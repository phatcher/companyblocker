from __future__ import annotations

import polars as pl

_VARIANT_SYSTEM_URI_COL = "system_uri"
_VARIANT_SOURCE_URI_COL = "source_uri"
_VARIANT_NAME_COL = "name"


def expand_target_frame_with_name_variants(
    target_frame: pl.DataFrame,
    variant_frame: pl.DataFrame | None,
    *,
    target_id_col: str,
    name_col: str,
) -> pl.DataFrame:
    """Add one extra row per explicit name-variant sidecar entry.

    Every added row maps back to the same `target_id_col` value, so
    `ClusteringStrategy.build_target_index()` indexes each known name variant as its own
    key resolving to that entity.

    No change to `build_target_index()`/`score_source_chunk()` themselves is
    needed for this: `TargetClusteringIndex.target_ids` is just a list
    aligned by position with `target_matrix`'s rows, with no uniqueness
    requirement -- multiple rows already legitimately resolve to the same
    target id today (see `target_index_lifecycle.build_target_index_from_frame`).
    This function's whole job is producing that expanded row set from a
    `*-names-*.parquet` sidecar frame (GLEIF's `previous`/`SUCCESSOR_CHAIN_PREDECESSOR`
    rows, Wikidata's `official`/`short`/`alias`/`label` rows, GB's `previous`
    rows) *before* the caller runs its usual
    tokenize/text-view/`build_target_index()` pipeline on the result -- every
    variant row flows through the exact same feature-construction path
    as the canonical name, so this works unchanged across every
    `representation` (`tfidf`/`wordpiece`/`sentencepiece`/`sbert`), not just
    one.

    Collision policy:

    - **Cross-entity collisions** (two different `target_id_col` values
      sharing an identical variant name string) are deliberately *not*
      deduplicated or resolved here -- each becomes its own index row/key
      resolving to its own target id, left for downstream candidate
      generation and pruning (`src/blocking/workflow.py`'s
      `_apply_similarity_ratio_pruning`/`_apply_target_cap`) to sort out like
      any other ambiguous match. This is a deliberate no-op, not an
      oversight: blocking tolerates false positives -- a spurious extra
      candidate costs one more downstream comparison, not correctness -- so
      there is no correctness reason to pick a winner at index-construction
      time, and picking one via some arbitrary rule (first-seen,
      alphabetical) would silently and non-deterministically discard a
      legitimate candidate key for whichever entity lost the tie.
    - **Same-entity redundant keys** (a variant row whose text is identical,
      case-insensitively, to that same entity's own canonical
      `target_frame[name_col]` value) *are* dropped. This isn't a
      correctness concern either -- it's a resource one: keeping a
      byte-identical duplicate key adds no new information (it can never
      shift the top-k candidates a source row would receive) but does spend
      one of the per-source `top_k`/`max_candidates_per_source` budget slots
      on a redundant self-match. GLEIF's `name_type == "primary"` rows in
      particular are exact duplicates of the canonical `name` column by
      construction (verified directly against real data: 0 mismatches
      across a 2,000-row sample), so without this filter every GLEIF-backed
      target index would silently double in size for zero informational
      gain.

    Args:
        target_frame: The base one-row-per-entity target frame (e.g. as
            returned by `blocking.loader.load_country_frame()`), *before*
            tokenization/text-view construction. Only `target_id_col` and
            `name_col` are read; every other column is preserved on the
            original rows and left null on appended variant rows (callers
            that build a target index only ever read `target_id_col`/
            `name_col` — see `ClusteringStrategy.build_target_index()`'s
            contract — so this is safe).
        variant_frame: A `*-names-*.parquet` sidecar frame with (at least)
            `"name"` and one of `"source_uri"`/`"system_uri"` columns, or
            `None`/empty when no sidecar exists for this target system (the
            common case outside GLEIF/Wikidata/GB) -- both are a no-op,
            returning `target_frame` unchanged.
        target_id_col: Column in `target_frame` holding target ids; matched
            against the primary entity's own identity in `variant_frame`.
        name_col: Column in `target_frame` holding the canonical name text;
            matched against `variant_frame["name"]` for the same-entity
            redundant-key filter, and the column appended variant rows carry
            their variant text under.

    Returns:
        `target_frame` unchanged if there is nothing to add, otherwise
        `target_frame` concatenated with one extra row per kept variant
        (schema-relaxed: columns other than `target_id_col`/`name_col` are
        null on the appended rows).
    """
    if variant_frame is None or variant_frame.height == 0 or target_frame.height == 0:
        return target_frame
    # GLEIF's/Wikidata's sidecar rows carry their own unique `system_uri`
    # plus a `source_uri` back-reference to the primary entity -- that
    # back-reference is what must match `target_id_col`, since `system_uri`
    # is not the primary's identity for those systems. GB's sidecar has no
    # `source_uri`, so its own `system_uri` (the primary's identity there)
    # is the fallback.
    primary_id_col = (
        _VARIANT_SOURCE_URI_COL
        if _VARIANT_SOURCE_URI_COL in variant_frame.columns
        else _VARIANT_SYSTEM_URI_COL
    )
    missing = {primary_id_col, _VARIANT_NAME_COL} - set(variant_frame.columns)
    if missing:
        raise ValueError(
            "variant_frame must have "
            f"'{primary_id_col}'/'{_VARIANT_NAME_COL}' columns (the "
            f"*-names-*.parquet sidecar schema); missing: {sorted(missing)}"
        )

    canonical_ids = target_frame.select(
        pl.col(target_id_col).cast(pl.Utf8).alias(target_id_col)
    ).unique()

    variant_rows = (
        variant_frame.lazy()
        .select(
            pl.col(primary_id_col).cast(pl.Utf8).alias(target_id_col),
            pl.col(_VARIANT_NAME_COL)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            .alias(name_col),
        )
        .filter(pl.col(name_col) != "")
        .join(canonical_ids.lazy(), on=target_id_col, how="inner")
        .unique()
        .collect()
    )
    if variant_rows.height == 0:
        return target_frame

    canonical_lookup = target_frame.select(
        pl.col(target_id_col).cast(pl.Utf8).alias(target_id_col),
        pl.col(name_col)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
        .alias("_canonical_upper"),
    )
    variant_rows = (
        variant_rows.join(canonical_lookup, on=target_id_col, how="left")
        .filter(pl.col(name_col).str.to_uppercase() != pl.col("_canonical_upper"))
        .drop("_canonical_upper")
        .unique()
    )
    if variant_rows.height == 0:
        return target_frame

    return pl.concat(
        [target_frame, variant_rows.select([target_id_col, name_col])],
        how="diagonal_relaxed",
    )
