from __future__ import annotations

import os
import random
import re
import time

import pytest
from company_cleanse.normalize import normalize_suffix_surface
from company_cleanse.rules import (
    _build_company_type_suffix_trie,
    _match_company_type_from_suffix_surface,
    get_company_type_rules,
)

pytestmark = pytest.mark.performance


def _normalize_clean_surface(value: str) -> str:
    normalized = (
        normalize_suffix_surface(
            value,
            and_tokens=("AND",),
            normalization_profile="default|-punctuation|punctuation_bang",
        )
        or ""
    )
    return re.sub(r"\s+", " ", normalized.replace("!", " ")).strip()


def _build_random_prefix(rng: random.Random) -> str:
    tokens = [
        "ALPHA",
        "BETA",
        "GAMMA",
        "DELTA",
        "OMEGA",
        "GLOBAL",
        "HOLDINGS",
        "SERVICES",
        "VENTURES",
        "GROUP",
        "NETWORK",
        "CAPITAL",
        "INDUSTRIES",
        "SYSTEMS",
        "TRADING",
        "SOLUTIONS",
        "DIGITAL",
        "PROJECT",
        "PARTNERS",
        "CONSULTING",
        "LOGISTICS",
        "EUROPE",
    ]
    length = rng.randint(2, 5)
    return " ".join(rng.choice(tokens) for _ in range(length))


def _build_non_match_suffixes() -> list[str]:
    # Chosen to be outside legal-form vocabulary and avoid accidental suffix matches.
    return [
        "NOTATYPE",
        "UNMAPPED ENTITY",
        "FANTASY STRUCTURE",
        "CUSTOM ORG FORM",
        "NO LEGAL SUFFIX",
        "UNKNOWN TAIL",
        "NONSTANDARD UNIT",
    ]


def _build_surfaces(
    rows: int, regex: str, mapping: dict[str, str], seed: int, non_match_ratio: float
) -> list[str]:
    rng = random.Random(seed)
    variants = sorted(mapping.keys())
    non_match_suffixes = _build_non_match_suffixes()
    non_match_target = max(0, min(rows, int(rows * non_match_ratio)))

    surfaces: list[str] = []

    # Match rows: real company-type variants sampled from the actual rule mapping.
    for _ in range(rows - non_match_target):
        suffix = rng.choice(variants)
        raw = f"{_build_random_prefix(rng)} {suffix}"
        surfaces.append(_normalize_clean_surface(raw))

    # Non-match rows: explicitly validate they do not match the regex path.
    non_matches_added = 0
    while non_matches_added < non_match_target:
        suffix = rng.choice(non_match_suffixes)
        raw = f"{_build_random_prefix(rng)} {suffix}"
        surface = _normalize_clean_surface(raw)
        if not re.search(regex, surface, re.IGNORECASE):
            surfaces.append(surface)
            non_matches_added += 1

    rng.shuffle(surfaces)
    return surfaces


@pytest.mark.skipif(
    os.environ.get("PERF_BENCHMARK", "0") != "1",
    reason="Set PERF_BENCHMARK=1 to run performance benchmarks.",
)
def test_company_type_matcher_regex_vs_trie_in_memory() -> None:
    rows = int(os.environ.get("PERF_ROWS", "120000"))
    seed = int(os.environ.get("PERF_SEED", "20260620"))
    non_match_ratio = float(os.environ.get("PERF_NON_MATCH_RATIO", "0.12"))
    regex, mapping = get_company_type_rules()
    # Guardrail: benchmark should exercise the full global rule set, not a single-country subset.
    assert "gmbh" in mapping
    assert "sp zoo" in mapping
    surfaces = _build_surfaces(
        rows, regex=regex, mapping=mapping, seed=seed, non_match_ratio=non_match_ratio
    )

    trie, max_tokens = _build_company_type_suffix_trie(mapping)

    t0 = time.perf_counter()
    regex_matches = [
        (match.group(1) if match else None)
        for match in (re.search(regex, surface, re.IGNORECASE) for surface in surfaces)
    ]
    regex_elapsed = max(time.perf_counter() - t0, 1e-9)

    t1 = time.perf_counter()
    trie_matches = [
        _match_company_type_from_suffix_surface(surface, trie, max_tokens)
        for surface in surfaces
    ]
    trie_elapsed = max(time.perf_counter() - t1, 1e-9)

    assert trie_matches == regex_matches

    regex_rps = len(surfaces) / regex_elapsed
    trie_rps = len(surfaces) / trie_elapsed
    print(
        "company_type matcher in-memory benchmark: "
        f"rows={len(surfaces):,}, "
        f"seed={seed}, non_match_ratio={non_match_ratio:.2f}, "
        f"regex={regex_rps:,.0f} rows/s ({regex_elapsed:.3f}s), "
        f"trie={trie_rps:,.0f} rows/s ({trie_elapsed:.3f}s), "
        f"speedup={(trie_rps / regex_rps):.3f}x"
    )
