from acquisition.models import ResearchPlan, SystemPlan
from acquisition.pipeline import normalize_systems
from acquisition.registry import COUNTRY_REGISTRY, SYSTEM_REGISTRY, get_system_plan


def test_normalize_systems_accepts_comma_and_space_separated_values():
    assert normalize_systems(["gleif,gb", " fr "]) == ["gleif", "gb", "fr"]


def test_normalize_systems_expands_all_keyword(monkeypatch):
    monkeypatch.setattr(
        "acquisition.pipeline._supported_system_codes", lambda: ["fr", "gb", "gleif"]
    )
    assert normalize_systems("all") == ["fr", "gb", "gleif"]


def test_registry_marks_current_support_levels():
    assert get_system_plan("fr").status == "live"
    assert get_system_plan("gb").status == "live"
    assert get_system_plan("de").status == "blocked"
    assert get_system_plan("es").status == "research_required"
    assert get_system_plan("nl").status == "research_required"
    assert get_system_plan("ie").status == "live"


def test_registry_exposes_nested_research_only_for_unsupported_systems():
    assert get_system_plan("fr").research is None
    assert get_system_plan("ie").research is None
    assert get_system_plan("de").research is not None
    assert get_system_plan("es").research is not None


def test_system_plan_validates_status_to_research_consistency():
    try:
        SystemPlan(
            code="xx",
            display_name="Example",
            region="eu",
            status="live",
            access_mode="bulk_download",
            info_url="https://example.com",
            notes="example",
            all_systems_target=True,
            research=ResearchPlan(basis="should_not_exist_for_live"),
        )
    except ValueError as exc:
        assert "live/supported systems must not define research metadata" in str(exc)
    else:
        raise AssertionError(
            "Expected supported plan with research metadata to be rejected"
        )

    try:
        SystemPlan(
            code="yy",
            display_name="Example Blocked",
            region="eu",
            status="blocked",
            access_mode="search_portal",
            info_url="https://example.com",
            notes="example",
            all_systems_target=False,
        )
    except ValueError as exc:
        assert "requires research metadata" in str(exc)
    else:
        raise AssertionError(
            "Expected blocked plan without research metadata to be rejected"
        )


def test_system_plan_rejects_invalid_status_or_access_mode():
    try:
        SystemPlan(
            code="zz",
            display_name="Invalid Status",
            region="eu",
            status="unknown",
            access_mode="bulk_download",
            info_url="https://example.com",
            notes="example",
            all_systems_target=False,
            research=ResearchPlan(),
        )
    except ValueError as exc:
        assert "invalid status" in str(exc)
    else:
        raise AssertionError("Expected invalid status to be rejected")

    try:
        SystemPlan(
            code="zx",
            display_name="Invalid Access",
            region="eu",
            status="live",
            access_mode="unknown",
            info_url="https://example.com",
            notes="example",
            all_systems_target=True,
        )
    except ValueError as exc:
        assert "invalid access_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid access_mode to be rejected")


def test_system_plan_rejects_all_membership_with_non_runnable_status():
    """`--systems all` may only sweep systems a run can proceed against, so
    the flag and the status cannot drift apart. `stopped` is the case
    that matters most: it is never runnable, with or without allow_research."""
    for status in ("research_required", "blocked", "stopped"):
        try:
            SystemPlan(
                code="xx",
                display_name="Flagged but not runnable",
                region="eu",
                status=status,
                access_mode="bulk_download",
                info_url="https://example.com",
                notes="example",
                all_systems_target=True,
                research=ResearchPlan(),
            )
        except ValueError as exc:
            assert (
                f"all_systems_target is set but status '{status}' is not runnable"
                in str(exc)
            )
        else:
            raise AssertionError(
                f"Expected '{status}' with all_systems_target to be rejected"
            )


def test_system_plan_accepts_all_membership_for_supported_not_only_live():
    """`supported` is runnable on its own, so a `supported` system may be in
    `all` -- membership is a separate decision from status, not a stricter
    reading of it."""
    plan = SystemPlan(
        code="xx",
        display_name="Supported and swept",
        region="eu",
        status="supported",
        access_mode="bulk_download",
        info_url="https://example.com",
        notes="example",
        all_systems_target=True,
    )

    assert plan.all_systems_target is True


def test_system_plan_rejects_non_boolean_all_systems_target():
    try:
        SystemPlan(
            code="xx",
            display_name="Bad flag",
            region="eu",
            status="live",
            access_mode="bulk_download",
            info_url="https://example.com",
            notes="example",
            all_systems_target="yes",  # type: ignore[arg-type]
        )
    except TypeError as exc:
        assert "all_systems_target must be a boolean, got str" in str(exc)
    else:
        raise AssertionError("Expected non-boolean all_systems_target to be rejected")


def test_research_plan_rejects_invalid_priority():
    try:
        ResearchPlan(priority="P9")
    except ValueError as exc:
        assert "Invalid research priority" in str(exc)
    else:
        raise AssertionError("Expected invalid research priority to be rejected")


def test_system_registry_includes_gleif_and_country_aliases():
    assert get_system_plan("gleif").status == "live"
    assert get_system_plan("gleif").code == "gleif"
    assert "gleif" in SYSTEM_REGISTRY
    assert get_system_plan("gb").code == "gb"


def test_registry_rejects_uk_alias_for_united_kingdom():
    try:
        get_system_plan("uk")
    except KeyError as exc:
        assert "Unknown system 'uk'" in str(exc)
    else:
        raise AssertionError("Expected uk to be rejected")


def test_registry_includes_all_eu27_iso_codes():
    eu27 = {
        "at",
        "be",
        "bg",
        "hr",
        "cy",
        "cz",
        "dk",
        "ee",
        "fi",
        "fr",
        "de",
        "gr",
        "hu",
        "ie",
        "it",
        "lv",
        "lt",
        "lu",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "sk",
        "si",
        "es",
        "se",
    }
    assert eu27.issubset(set(COUNTRY_REGISTRY.keys()))
