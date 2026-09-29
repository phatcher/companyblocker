import polars as pl
import pytest

from blocking.name_transform import (
    NAME_TRANSFORMS,
    DerivedColumnTransform,
    IdentityTransform,
    NameTransform,
    derive_name_forms,
    resolve_name_transform,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "system_uri": ["a", "b", "c"],
            "name": ["Acme Limited", "ACME LTD", "International Business Machines"],
            # Stale values a loaded cleansed layer might carry: the derivation
            # must overwrite them, not trust them.
            "name_cleansed": ["stale", "stale", "stale"],
        }
    )


def test_derive_name_forms_replaces_whatever_the_frame_carried() -> None:
    """Two spellings that differ raw and agree once cleansed come out equal,
    and the stale column the layer carried is gone."""
    out = derive_name_forms(_frame(), profile="default")

    assert out.get_column("name_cleansed").to_list()[:2] == ["acme ltd", "acme ltd"]
    assert "stale" not in out.get_column("name_cleansed").to_list()
    assert out.get_column("short_name").to_list()[:2] == ["acme", "acme"]
    assert set(out.columns) >= {"name_cleansed_basic", "short_name", "acronym"}


def test_derive_name_forms_two_profiles_give_two_views_of_one_name() -> None:
    """The profile is a per-run parameter, so two runs differing only in it
    derive different forms of the same name."""
    lowercased = derive_name_forms(_frame(), profile="default")
    cased = derive_name_forms(_frame(), profile="default|-lowercase")

    assert lowercased.get_column("name_cleansed")[0] == "acme ltd"
    assert cased.get_column("name_cleansed")[0] != "acme ltd"
    assert cased.get_column("name_cleansed")[0].startswith("Acme")


def test_derive_name_forms_refuses_a_frame_without_a_raw_name() -> None:
    with pytest.raises(ValueError, match="'name'"):
        derive_name_forms(pl.DataFrame({"system_uri": ["a"]}), profile="default")


def test_identity_scores_the_column_each_side_always_read() -> None:
    transform = resolve_name_transform("identity")

    assert transform == IdentityTransform()
    assert transform.apply(_frame()).equals(_frame())
    assert transform.scored_column("name") == "name"
    assert transform.scored_column("short_name") == "short_name"
    assert transform.describe() == {"kind": "identity", "column": "name"}


def test_cleanse_short_name_and_acronym_score_their_derived_columns() -> None:
    """The choice between them is which derived column the run scores, not
    which function ran: every form is derived once, before the choice."""
    derived = derive_name_forms(_frame(), profile="default")

    for name, column in (
        ("cleanse", "name_cleansed"),
        ("short_name", "short_name"),
        ("acronym", "acronym"),
    ):
        transform = resolve_name_transform(name)
        assert transform == DerivedColumnTransform(name, column)
        assert transform.scored_column("name") == column
        assert transform.apply(derived).equals(derived)
        assert transform.describe() == {"kind": name, "column": column}


def test_a_derived_column_transform_refuses_an_underived_frame() -> None:
    with pytest.raises(ValueError, match="derive_name_forms"):
        resolve_name_transform("acronym").apply(pl.DataFrame({"name": ["Acme"]}))


def test_resolve_name_transform_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="name_transform must be one of"):
        resolve_name_transform("uppercase")


def test_every_registered_transform_satisfies_the_protocol() -> None:
    for name in NAME_TRANSFORMS:
        assert isinstance(resolve_name_transform(name), NameTransform), name
