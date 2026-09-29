from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from acquisition import canonical
from acquisition.constants_status import STATUS_RESEARCH_REQUIRED
from acquisition.registry import get_system_plan
from workspace.roots import WorkspaceRoots


def test_canonicalize_system_primary_name_override_promotes_best_transliteration(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_primary_name_override

    canonical_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gleif://A", "gleif://B", "gleif://C"],
            "name": [
                "マルチチュード",
                "Alpha Ltd",
                'מאלטיטיוד בע"מ',
            ],
        }
    ).write_parquet(canonical_dir / "gleif-001.parquet")

    pl.DataFrame(
        {
            "system_uri": ["gleif://A", "gleif://A", "gleif://C"],
            "LEI": ["A", "A", "C"],
            "name": [
                "MULTITUDE JAPAN AUTO",
                "MULTITUDE JAPAN PREFERRED",
                "MULTITUDE LTD",
            ],
            "source_type": [
                "AUTO_ASCII_TRANSLITERATED_LEGAL_NAME",
                "PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME",
                "PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME",
            ],
            "language_code": [None, None, None],
            "derivation_note": ["native TransliteratedOtherEntityName"] * 3,
            "name_type": [
                "transliteration_auto",
                "transliteration_preferred",
                "transliteration_preferred",
            ],
        }
    ).write_parquet(canonical_dir / "gleif-names-001.parquet")

    written = canonicalize_system_primary_name_override(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    assert [path.name for path in written] == ["gleif-001.parquet"]
    out = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    names_by_uri = dict(zip(out["system_uri"], out["name"], strict=True))
    # Entity A has both an auto and a preferred transliteration -- preferred wins.
    assert names_by_uri["gleif://A"] == "MULTITUDE JAPAN PREFERRED"
    # Entity B's name is already Latin-script -- left untouched.
    assert names_by_uri["gleif://B"] == "Alpha Ltd"
    # Entity C's Hebrew legal name is replaced by its sole transliteration.
    assert names_by_uri["gleif://C"] == "MULTITUDE LTD"


def test_canonicalize_system_primary_name_override_rewrites_a_partitioned_snapshot(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """Companion to the flat-layout case above: this step rewrites
    canonical entity files in place, so it has to find them under
    `jurisdiction_code=*/`. A resolver that missed them would leave every
    non-Latin name unpromoted and report success.
    """
    from acquisition.canonical import canonicalize_system_primary_name_override

    canonical_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    for jurisdiction, uri, name in [
        ("jp", "gleif://A", "マルチチュード"),
        ("il", "gleif://C", 'מאלטיטיוד בע"מ'),
    ]:
        partition_dir = canonical_dir / f"jurisdiction_code={jurisdiction}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"system_uri": [uri], "name": [name]}).write_parquet(
            partition_dir / "part-00001.parquet"
        )

    pl.DataFrame(
        {
            "system_uri": ["gleif://A", "gleif://C"],
            "LEI": ["A", "C"],
            "name": ["MULTITUDE JAPAN PREFERRED", "MULTITUDE LTD"],
            "source_type": ["PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME"] * 2,
            "language_code": [None, None],
            "derivation_note": ["native TransliteratedOtherEntityName"] * 2,
            "name_type": ["transliteration_preferred"] * 2,
        },
        schema_overrides={"language_code": pl.Utf8},
    ).write_parquet(canonical_dir / "gleif-names-001.parquet")

    written = canonicalize_system_primary_name_override(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    # Both partitions rewritten, not just whichever one sorted first.
    assert len(written) == 2
    promoted = {
        row["system_uri"]: row["name"]
        for row in pl.read_parquet(written).iter_rows(named=True)
    }
    assert promoted["gleif://A"] == "MULTITUDE JAPAN PREFERRED"
    assert promoted["gleif://C"] == "MULTITUDE LTD"


def test_canonicalize_system_primary_name_override_ignores_non_transliteration_types(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    """A `previous`/`trading`/`alternative_language` row is a different name,
    not a script rendering of the current one. If a catalog author mistakenly
    lists one in primary_name_override_type_priority, it must be silently
    ignored -- never applied, and never a crash -- while a genuine
    transliteration type in the same list still works normally.
    """
    from acquisition.canonical import canonicalize_system_primary_name_override

    canonical_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gleif://A"],
            "name": ["マルチチュード"],
        }
    ).write_parquet(canonical_dir / "gleif-001.parquet")

    pl.DataFrame(
        {
            "system_uri": ["gleif://A", "gleif://A"],
            "LEI": ["A", "A"],
            "name": ["MULTITUDE OLD PREVIOUS NAME", "MULTITUDE JAPAN PREFERRED"],
            "source_type": [
                "PREVIOUS_LEGAL_NAME",
                "PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME",
            ],
            "language_code": [None, None],
            "derivation_note": [
                "native OtherEntityName",
                "native TransliteratedOtherEntityName",
            ],
            "name_type": ["previous", "transliteration_preferred"],
        }
    ).write_parquet(canonical_dir / "gleif-names-001.parquet")

    plan = get_system_plan("gleif")
    mocker.patch.object(
        canonical,
        "get_system_plan",
        return_value=replace(
            plan,
            primary_name_override_type_priority=(
                "previous",
                "transliteration_preferred",
            ),
        ),
    )

    canonicalize_system_primary_name_override(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    out = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    assert out["name"][0] == "MULTITUDE JAPAN PREFERRED"


def test_canonicalize_system_primary_name_override_is_a_noop_without_priority_config(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_primary_name_override

    canonical_dir = layer_fixture_dir("gb", layer="canonical") / "2026-06-01"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    assert (
        canonicalize_system_primary_name_override(
            "gb", run_date="2026-06-15", roots=workspace_roots
        )
        == []
    )


def test_canonicalize_system_primary_name_override_research_policy_denies_stage(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import canonicalize_system_primary_name_override

    denied_plan = SimpleNamespace(
        code="gleif",
        status=STATUS_RESEARCH_REQUIRED,
        notes="research pending",
        primary_name_override_type_priority=("transliteration_preferred",),
        research=SimpleNamespace(
            allow_research_runtime=False, allowed_stages=("shard",)
        ),
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=denied_plan
    )

    with pytest.raises(
        RuntimeError, match="research execution policy denies stage 'canonical'"
    ):
        canonicalize_system_primary_name_override(
            "gleif", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("gleif")


def test_canonicalize_system_primary_name_override_blocked_status_rejected_even_with_allow_research(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import canonicalize_system_primary_name_override

    blocked_plan = SimpleNamespace(
        code="gleif",
        status="blocked",
        notes="No access",
        primary_name_override_type_priority=("transliteration_preferred",),
        research=SimpleNamespace(
            allow_research_runtime=True, allowed_stages=("canonical",)
        ),
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=blocked_plan
    )

    with pytest.raises(RuntimeError, match="gleif: blocked - No access"):
        canonicalize_system_primary_name_override(
            "gleif", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("gleif")
