from __future__ import annotations

from typing import Protocol, TypedDict

# System plan statuses
STATUS_LIVE = "live"
STATUS_SUPPORTED = "supported"
STATUS_RESEARCH_REQUIRED = "research_required"
STATUS_BLOCKED = "blocked"
STATUS_STOPPED = "stopped"
"""No longer supported due to other factors. Not runnable even with
allow_research (excluded from every runnable-status set this module
defines) -- reclassify the system's status in its catalog entry to make it
runnable again."""

ALLOWED_SYSTEM_STATUSES = frozenset(
    {
        STATUS_LIVE,
        STATUS_SUPPORTED,
        STATUS_RESEARCH_REQUIRED,
        STATUS_BLOCKED,
        STATUS_STOPPED,
    }
)

RUNNABLE_SYSTEM_STATUSES = frozenset(
    {
        STATUS_LIVE,
        STATUS_SUPPORTED,
    }
)


def resolve_runnable_statuses(allow_research: bool) -> frozenset[str]:
    """Return the system statuses a run may proceed against.

    Always includes `RUNNABLE_SYSTEM_STATUSES` (`live`/`supported`); also
    includes `STATUS_RESEARCH_REQUIRED` when `allow_research` is set, since
    that's the opt-in that unlocks a `research_required` system for a run.
    `STATUS_BLOCKED`/`STATUS_STOPPED` are never included, with or without
    `allow_research`.
    """
    if allow_research:
        return RUNNABLE_SYSTEM_STATUSES | {STATUS_RESEARCH_REQUIRED}
    return RUNNABLE_SYSTEM_STATUSES


def is_research_stage_allowed(plan, *, stage: str) -> bool:
    """Whether `plan`'s research execution policy opts `stage` in.

    False for any plan with no `research` policy at all, with
    `allow_research_runtime` off, or whose `allowed_stages` omits `stage` --
    a missing or malformed policy is never treated as permission.
    """
    research = getattr(plan, "research", None)
    if research is None:
        return False
    if not getattr(research, "allow_research_runtime", False):
        return False
    allowed_stages = set(getattr(research, "allowed_stages", ()) or ())
    return stage in allowed_stages


def require_runnable_plan(plan, *, allow_research: bool, stage: str) -> None:
    """Raise unless `plan` may be run for `stage`, or return None.

    Two gates, in this order: the plan's status must be in
    `resolve_runnable_statuses(allow_research)`, and a `research_required`
    plan unlocked by `allow_research` must additionally name `stage` in its
    own research execution policy.

    Every stage entry point runs both -- call this rather than open-coding
    them, so a new entry point cannot ship with the second gate missing and
    silently run a research system the catalog opted out of that stage.
    """
    if plan.status not in resolve_runnable_statuses(allow_research):
        raise RuntimeError(f"{plan.code}: {plan.status} - {plan.notes}")

    if (
        plan.status == STATUS_RESEARCH_REQUIRED
        and allow_research
        and not is_research_stage_allowed(plan, stage=stage)
    ):
        raise RuntimeError(
            f"{plan.code}: research execution policy denies stage '{stage}'. "
            f"Set research.allow_research_runtime=true and include '{stage}' in "
            "research.allowed_stages."
        )


SUPPORTED_COUNTRY_SELECTOR_LIVE = "live"
SUPPORTED_COUNTRY_SELECTOR_SUPPORTED = "supported"
SUPPORTED_COUNTRY_SELECTOR_LOCAL = "local"

ALLOWED_SUPPORTED_COUNTRY_SELECTORS = frozenset(
    {
        SUPPORTED_COUNTRY_SELECTOR_LIVE,
        SUPPORTED_COUNTRY_SELECTOR_SUPPORTED,
        SUPPORTED_COUNTRY_SELECTOR_LOCAL,
    }
)


class ProgressReporter(Protocol):
    def __call__(self, message: str) -> None: ...


class AcquisitionStatusPayload(TypedDict, total=False):
    status: str
    message: str
    artifact_count: int
    freshness_gate_applied: bool
    freshness_gate_passed: bool
    freshness_gate_reason: str


__all__ = [
    "ALLOWED_SUPPORTED_COUNTRY_SELECTORS",
    "ALLOWED_SYSTEM_STATUSES",
    "AcquisitionStatusPayload",
    "ProgressReporter",
    "RUNNABLE_SYSTEM_STATUSES",
    "STATUS_BLOCKED",
    "STATUS_LIVE",
    "STATUS_RESEARCH_REQUIRED",
    "STATUS_STOPPED",
    "STATUS_SUPPORTED",
    "SUPPORTED_COUNTRY_SELECTOR_LIVE",
    "SUPPORTED_COUNTRY_SELECTOR_LOCAL",
    "SUPPORTED_COUNTRY_SELECTOR_SUPPORTED",
    "is_research_stage_allowed",
    "require_runnable_plan",
    "resolve_runnable_statuses",
]
