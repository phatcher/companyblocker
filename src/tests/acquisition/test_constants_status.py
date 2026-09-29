from types import SimpleNamespace

import pytest

from acquisition.constants_status import (
    RUNNABLE_SYSTEM_STATUSES,
    STATUS_BLOCKED,
    STATUS_LIVE,
    STATUS_RESEARCH_REQUIRED,
    STATUS_STOPPED,
    STATUS_SUPPORTED,
    is_research_stage_allowed,
    require_runnable_plan,
    resolve_runnable_statuses,
)


def _plan(status: str, *, research=None):
    return SimpleNamespace(
        code="xx", status=status, notes="test plan", research=research
    )


def test_resolve_runnable_statuses_without_allow_research_matches_base_set() -> None:
    assert resolve_runnable_statuses(allow_research=False) == RUNNABLE_SYSTEM_STATUSES


def test_resolve_runnable_statuses_with_allow_research_adds_research_required() -> None:
    resolved = resolve_runnable_statuses(allow_research=True)
    assert resolved == RUNNABLE_SYSTEM_STATUSES | {STATUS_RESEARCH_REQUIRED}
    assert STATUS_LIVE in resolved
    assert STATUS_SUPPORTED in resolved
    assert STATUS_RESEARCH_REQUIRED in resolved


def test_resolve_runnable_statuses_never_includes_blocked_or_stopped() -> None:
    for allow_research in (False, True):
        resolved = resolve_runnable_statuses(allow_research=allow_research)
        assert STATUS_BLOCKED not in resolved
        assert STATUS_STOPPED not in resolved


def test_is_research_stage_allowed_rejects_missing_or_disabled_policy() -> None:
    assert is_research_stage_allowed(_plan(STATUS_LIVE), stage="canonical") is False
    assert (
        is_research_stage_allowed(
            _plan(
                STATUS_LIVE,
                research=SimpleNamespace(
                    allow_research_runtime=False, allowed_stages=("canonical",)
                ),
            ),
            stage="canonical",
        )
        is False
    )


def test_is_research_stage_allowed_requires_the_stage_to_be_listed() -> None:
    research = SimpleNamespace(
        allow_research_runtime=True, allowed_stages=("canonical",)
    )
    assert (
        is_research_stage_allowed(_plan(STATUS_LIVE, research=research), stage="shard")
        is False
    )
    assert (
        is_research_stage_allowed(
            _plan(STATUS_LIVE, research=research), stage="canonical"
        )
        is True
    )


def test_require_runnable_plan_accepts_a_runnable_status() -> None:
    require_runnable_plan(_plan(STATUS_LIVE), allow_research=False, stage="canonical")


def test_require_runnable_plan_rejects_a_non_runnable_status() -> None:
    with pytest.raises(RuntimeError, match="xx: blocked - test plan"):
        require_runnable_plan(
            _plan(STATUS_BLOCKED), allow_research=True, stage="canonical"
        )


def test_require_runnable_plan_rejects_research_status_without_allow_research() -> None:
    with pytest.raises(RuntimeError, match="xx: research_required - test plan"):
        require_runnable_plan(
            _plan(STATUS_RESEARCH_REQUIRED), allow_research=False, stage="canonical"
        )


def test_require_runnable_plan_rejects_a_stage_the_research_policy_omits() -> None:
    plan = _plan(
        STATUS_RESEARCH_REQUIRED,
        research=SimpleNamespace(
            allow_research_runtime=True, allowed_stages=("shard",)
        ),
    )
    with pytest.raises(
        RuntimeError, match="research execution policy denies stage 'canonical'"
    ):
        require_runnable_plan(plan, allow_research=True, stage="canonical")

    require_runnable_plan(plan, allow_research=True, stage="shard")
