import polars as pl
import pytest
from company_vectorize import expand_target_frame_with_name_variants


def _target_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_expand_returns_unchanged_when_variant_frame_is_none():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])

    result = expand_target_frame_with_name_variants(
        target, None, target_id_col="system_uri", name_col="name"
    )

    assert result.equals(target)


def test_expand_returns_unchanged_when_variant_frame_is_empty():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])
    variants = pl.DataFrame(schema={"system_uri": pl.Utf8, "name": pl.Utf8})

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.equals(target)


def test_expand_returns_unchanged_when_target_frame_is_empty():
    target = pl.DataFrame(schema={"system_uri": pl.Utf8, "name": pl.Utf8})
    variants = pl.DataFrame(
        {"system_uri": ["t:1"], "name": ["Acme Holdings"], "name_type": ["previous"]}
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.equals(target)


def test_expand_adds_one_row_per_variant_resolving_to_same_target_id():
    target = _target_frame(
        [
            {"system_uri": "t:1", "name": "International Business Machines"},
            {"system_uri": "t:2", "name": "Zephyr Industries"},
        ]
    )
    variants = pl.DataFrame(
        {
            "system_uri": ["t:1", "t:1"],
            "name": ["IBM", "Big Blue"],
            "name_type": ["alias", "alias"],
        }
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.height == 4
    by_id = result.group_by("system_uri").agg(pl.col("name")).sort("system_uri")
    t1_names = set(by_id.filter(pl.col("system_uri") == "t:1")["name"][0])
    assert t1_names == {"International Business Machines", "IBM", "Big Blue"}
    t2_names = set(by_id.filter(pl.col("system_uri") == "t:2")["name"][0])
    assert t2_names == {"Zephyr Industries"}


def test_expand_drops_variant_identical_to_canonical_name_case_insensitive():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])
    variants = pl.DataFrame(
        {
            "system_uri": ["t:1", "t:1"],
            "name": ["ACME LTD", "Acme Holdings"],
            "name_type": ["primary", "previous"],
        }
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    names = sorted(result["name"].to_list())
    assert names == ["Acme Holdings", "Acme Ltd"]


def test_expand_drops_blank_variant_rows():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])
    variants = pl.DataFrame(
        {"system_uri": ["t:1", "t:1"], "name": ["   ", None], "name_type": ["a", "b"]}
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.equals(target)


def test_expand_ignores_variants_for_unknown_target_ids():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])
    variants = pl.DataFrame(
        {
            "system_uri": ["t:99"],
            "name": ["Unrelated Co"],
            "name_type": ["previous"],
        }
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.equals(target)


def test_expand_cross_entity_collision_keeps_both_as_separate_keys():
    """Collision policy: two different target entities sharing an identical
    variant name string both keep it as their own index key -- no
    dedup/tie-break at construction time. The ambiguity is left for
    downstream candidate generation and pruning to resolve, per
    `expand_target_frame_with_name_variants`'s documented collision policy."""
    target = _target_frame(
        [
            {"system_uri": "t:1", "name": "Acme Trading Co"},
            {"system_uri": "t:2", "name": "Acme Holdings Group"},
        ]
    )
    variants = pl.DataFrame(
        {
            "system_uri": ["t:1", "t:2"],
            "name": ["ACME", "ACME"],
            "name_type": ["trading", "trading"],
        }
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    acme_rows = result.filter(pl.col("name") == "ACME")
    assert acme_rows.height == 2
    assert set(acme_rows["system_uri"].to_list()) == {"t:1", "t:2"}


def test_expand_deduplicates_identical_variant_rows_for_same_entity():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])
    variants = pl.DataFrame(
        {
            "system_uri": ["t:1", "t:1"],
            "name": ["Acme Holdings", "Acme Holdings"],
            "name_type": ["previous", "previous"],
        }
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.height == 2


def test_expand_preserves_other_target_frame_columns_and_nulls_them_for_variants():
    target = _target_frame(
        [{"system_uri": "t:1", "name": "Acme Ltd", "jurisdiction_code": "gb"}]
    )
    variants = pl.DataFrame(
        {"system_uri": ["t:1"], "name": ["Acme Holdings"], "name_type": ["previous"]}
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.height == 2
    variant_row = result.filter(pl.col("name") == "Acme Holdings").row(0, named=True)
    assert variant_row["jurisdiction_code"] is None


def test_expand_raises_on_variant_frame_missing_required_columns():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])
    variants = pl.DataFrame({"lei": ["x"], "name": ["Acme Holdings"]})

    with pytest.raises(ValueError, match="system_uri"):
        expand_target_frame_with_name_variants(
            target, variants, target_id_col="system_uri", name_col="name"
        )


def test_expand_joins_on_source_uri_when_present_names_sidecar_shape():
    """The shape GLEIF and Wikidata name sidecars carry: each variant row
    holds its own unique `system_uri` (a `name://<hash>`) plus a `source_uri`
    back-reference to the primary entity, so the join must use `source_uri`
    and not the row's own unrelated `system_uri`."""
    target = _target_frame(
        [{"system_uri": "t:1", "name": "International Business Machines"}]
    )
    variants = pl.DataFrame(
        {
            "system_uri": ["name://aaaa000000000001", "name://aaaa000000000002"],
            "source_uri": ["t:1", "t:1"],
            "name": ["IBM", "Big Blue"],
            "name_type": ["alias", "alias"],
            "match_uri": [None, None],
        }
    )

    result = expand_target_frame_with_name_variants(
        target, variants, target_id_col="system_uri", name_col="name"
    )

    assert result.height == 3
    names = set(result["name"].to_list())
    assert names == {"International Business Machines", "IBM", "Big Blue"}
    assert set(result["system_uri"].to_list()) == {"t:1"}


def test_expand_raises_on_variant_frame_with_source_uri_but_missing_name():
    target = _target_frame([{"system_uri": "t:1", "name": "Acme Ltd"}])
    variants = pl.DataFrame({"system_uri": ["name://x"], "source_uri": ["t:1"]})

    with pytest.raises(ValueError, match="source_uri"):
        expand_target_frame_with_name_variants(
            target, variants, target_id_col="system_uri", name_col="name"
        )
