from __future__ import annotations

import json
from functools import cache
from pathlib import Path

# Kept acquisition-local: DE (gleif + offeneregister) is currently the only
# consumer. Promote to company_cleanse (same interface, swap pathlib for
# importlib.resources) if a second consumer -- especially one outside this
# repo -- needs authority-scoped identifier normalization.

_RESOURCES_DIR = Path(__file__).with_name("resources") / "authority_crosswalks"


@cache
def _load_crosswalk(scheme: str) -> dict[str, str]:
    path = _RESOURCES_DIR / f"{scheme}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_crosswalk(scheme: str) -> dict[str, str]:
    """Return the raw-authority-code -> canonical-authority-code mapping for a scheme.

    ``scheme`` names the *source's own* representation of an authority/court
    (for example ``"xjustiz"`` for offeneregister's court-code prefixes). The
    canonical value on the other side of the mapping is a GLEIF registration
    authority code, reused as the shared vocabulary since it already exists
    and is independently maintained.
    """
    return _load_crosswalk(scheme)


def resolve_authority_code(scheme: str, raw_authority_code: str | None) -> str | None:
    """Resolve a source-specific authority/court code to its canonical RA code.

    Returns ``None`` when ``raw_authority_code`` is empty or unmapped -- an
    unresolved authority must null out the identifier it scopes rather than
    fall back to an unscoped value, since a bare locally-issued number can
    collide across authorities.
    """
    if not raw_authority_code:
        return None
    return get_crosswalk(scheme).get(raw_authority_code)
