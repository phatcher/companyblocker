from __future__ import annotations

import polars as pl
import pytest

from acquisition.name_variant_uri import (
    NameVariantUniquenessError,
    check_name_variant_uniqueness,
)


def test_check_name_variant_uniqueness_passes_on_distinct_triples():
    frame = pl.DataFrame(
        {
            "source_uri": ["gleif://A", "gleif://A", "gleif://B"],
            "name_type": ["previous", "trading", "previous"],
            "name": ["Old Co", "Old Co", "Old Co"],
        }
    )
    check_name_variant_uniqueness(frame, system_code="gleif")


def test_check_name_variant_uniqueness_allows_same_name_across_different_entities():
    """Different entities legitimately sharing a former/trading name over
    time is real and must stay valid -- uniqueness is scoped per source_uri,
    not global."""
    frame = pl.DataFrame(
        {
            "source_uri": ["gleif://A", "gleif://B"],
            "name_type": ["previous", "previous"],
            "name": ["Old Co", "Old Co"],
        }
    )
    check_name_variant_uniqueness(frame, system_code="gleif")


def test_check_name_variant_uniqueness_rejects_genuine_duplicate():
    frame = pl.DataFrame(
        {
            "source_uri": ["gleif://A", "gleif://A"],
            "name_type": ["previous", "previous"],
            "name": ["Old Co", "Old Co"],
        }
    )
    with pytest.raises(NameVariantUniquenessError, match="gleif://A"):
        check_name_variant_uniqueness(frame, system_code="gleif")


def test_check_name_variant_uniqueness_aggregates_every_violation():
    frame = pl.DataFrame(
        {
            "source_uri": ["gleif://A", "gleif://A", "gleif://B", "gleif://B"],
            "name_type": ["previous", "previous", "trading", "trading"],
            "name": ["Old Co", "Old Co", "Trade Co", "Trade Co"],
        }
    )
    with pytest.raises(NameVariantUniquenessError) as exc_info:
        check_name_variant_uniqueness(frame, system_code="gleif")
    message = str(exc_info.value)
    assert "gleif://A" in message
    assert "gleif://B" in message
    assert "2 duplicate" in message
