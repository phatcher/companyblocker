from __future__ import annotations


def coerce_country_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        country = str(key).strip().lower()
        if not country:
            continue
        try:
            out[country] = int(value)
        except (TypeError, ValueError):
            continue
    return out
