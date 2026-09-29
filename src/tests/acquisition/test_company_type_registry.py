from __future__ import annotations

from types import SimpleNamespace

import pytest

from acquisition import company_type_registry
from acquisition.company_type_registry import get_system_company_type_mapping


def test_get_system_company_type_mapping_returns_empty_dict_when_no_column(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        company_type_registry,
        "get_system_plan",
        lambda system_code: SimpleNamespace(code=system_code, company_type_column=None),
    )
    monkeypatch.setattr(
        company_type_registry,
        "get_company_type_mapping",
        lambda code: (_ for _ in ()).throw(
            AssertionError("get_company_type_mapping should not be called")
        ),
    )

    assert get_system_company_type_mapping("perturbed") == {}


def test_get_system_company_type_mapping_delegates_to_mapping_lookup_by_plan_code(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        company_type_registry,
        "get_system_plan",
        lambda system_code: SimpleNamespace(
            code="resolved-code", company_type_column="CompanyCategory"
        ),
    )
    seen: dict[str, str] = {}

    def fake_get_company_type_mapping(code: str) -> dict[str, str]:
        seen["code"] = code
        return {"LTD": "Limited"}

    monkeypatch.setattr(
        company_type_registry, "get_company_type_mapping", fake_get_company_type_mapping
    )

    result = get_system_company_type_mapping("gb")

    assert result == {"LTD": "Limited"}
    assert seen == {"code": "resolved-code"}
