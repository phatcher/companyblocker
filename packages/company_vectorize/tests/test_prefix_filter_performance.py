"""Runtime/recall measurement for the opt-in prefix-filter pre-scoring step,
against the unfiltered `"sklearn"` backend baseline.

Not run by default CI (`-m "not performance"`, see `run_tests_ci.py`); run
explicitly with `pytest -m performance -s packages/company_vectorize/tests/
test_prefix_filter_performance.py` to reproduce the prefix filter's timing
numbers.

Dataset: synthetic but realistic company names -- each target row gets one
globally-unique "brand" token (the way a real company name usually carries at
least one distinctive term: a founder's surname, a coined brand word) plus 0-2
common industry words drawn from a small shared pool and a legal-form suffix,
mirroring the shape already used by this package's own fixtures (see
`test_sparse_similarity.py`'s "acme limited"/"acme holdings"/"beta partners").
Each source row is a perturbed variant of exactly one target row (token drop,
token-order shuffle, single-character typo), giving a known ground-truth match
per source row without needing real acquisition data under `data/`.
"""

from __future__ import annotations

import random
import time
from typing import Any

import polars as pl
import pytest
from company_vectorize import resolve_target_index_build_settings
from company_vectorize.sparse_similarity import (
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy

pytestmark = pytest.mark.performance

_COMMON_WORDS = [
    "global",
    "systems",
    "solutions",
    "holdings",
    "partners",
    "capital",
    "ventures",
    "industries",
    "technologies",
    "consulting",
    "logistics",
    "materials",
    "energy",
    "media",
    "financial",
    "insurance",
    "trading",
    "network",
    "digital",
    "atlantic",
    "pacific",
    "continental",
    "european",
    "northern",
    "southern",
    "eastern",
    "western",
    "united",
    "national",
    "regional",
    "international",
    "advanced",
    "precision",
    "dynamic",
    "strategic",
    "premier",
    "summit",
    "pioneer",
    "vertex",
    "horizon",
    "meridian",
    "cascade",
    "sterling",
    "quantum",
    "apex",
    "crown",
    "harbor",
    "group",
]
_LEGAL_SUFFIXES = [
    "limited",
    "ltd",
    "inc",
    "llc",
    "plc",
    "gmbh",
    "corp",
    "group",
    "holdings",
]
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _build_unique_token(rng: random.Random) -> str:
    length = rng.randint(5, 9)
    return "".join(rng.choice(_LETTERS) for _ in range(length))


def _build_target_name(rng: random.Random, unique_token: str) -> str:
    common = rng.sample(_COMMON_WORDS, rng.randint(0, 2))
    suffix = rng.choice(_LEGAL_SUFFIXES)
    tokens = [unique_token, *common, suffix]
    rng.shuffle(tokens)
    return " ".join(tokens)


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    pos = rng.randrange(1, len(word) - 1)
    return word[:pos] + rng.choice(_LETTERS) + word[pos + 1 :]


def _perturb(name: str, rng: random.Random) -> str:
    tokens = name.split()
    if rng.random() < 0.3 and len(tokens) > 1:
        drop_candidates = [
            i
            for i, t in enumerate(tokens)
            if t in _LEGAL_SUFFIXES or t in _COMMON_WORDS
        ]
        if drop_candidates:
            del tokens[rng.choice(drop_candidates)]
    if len(tokens) > 2 and rng.random() < 0.3:
        i, j = rng.sample(range(len(tokens)), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    if rng.random() < 0.4 and tokens:
        idx = rng.randrange(len(tokens))
        tokens[idx] = _typo(tokens[idx], rng)
    return " ".join(tokens)


def _build_dataset(n_rows: int, seed: int):
    rng = random.Random(seed)
    target_ids = [f"t{i}" for i in range(n_rows)]
    target_names = [
        _build_target_name(rng, _build_unique_token(rng)) for _ in range(n_rows)
    ]
    source_ids = [f"s{i}" for i in range(n_rows)]
    source_names = [_perturb(name, rng) for name in target_names]
    return target_ids, target_names, source_ids, source_names


def _recall_at_k(
    rows: list[dict[str, Any]],
    true_target_by_source: dict[str, str],
    k: int,
) -> float:
    by_source: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(
            (row["rank"], row["target_id"])
        )
    hits = 0
    for src_id, true_target in true_target_by_source.items():
        candidates = sorted(by_source.get(src_id, []), key=lambda rc: rc[0])
        if true_target in [target_id for _, target_id in candidates[:k]]:
            hits += 1
    return hits / len(true_target_by_source) if true_target_by_source else 0.0


def _run_backend(
    target_index,
    src_matrix,
    source_ids: list[str],
    true_target_by_source: dict[str, str],
    *,
    top_k: int,
    backend_options: dict[str, object] | None,
) -> dict[str, float]:
    t0 = time.perf_counter()
    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="sklearn",
        top_k=top_k,
        max_candidates_per_source=None,
        backend_options=backend_options,
        min_similarity=0.0,
    )
    build_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    rows = score_source_with_backend(
        source_ids,
        src_matrix,
        target_index=target_index,
        top_k=top_k,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=backend_index,
    )
    score_seconds = time.perf_counter() - t1

    return {
        "build_seconds": build_seconds,
        "score_seconds": score_seconds,
        "recall_at_k": _recall_at_k(rows, true_target_by_source, top_k),
    }


@pytest.mark.integration
@pytest.mark.parametrize("n_rows", [1000, 5000])
def test_prefix_filter_vs_unfiltered_baseline(n_rows: int) -> None:
    top_k = 5

    target_ids, target_names, source_ids, source_names = _build_dataset(
        n_rows, seed=20260828
    )
    true_target_by_source = dict(zip(source_ids, target_ids, strict=True))

    strategy = TfidfClusteringStrategy()
    target_frame = pl.DataFrame({"system_uri": target_ids, "name": target_names})
    target_index = strategy.build_target_index(
        target_frame,
        target_id_col="system_uri",
        text_col="name",
        build_settings=resolve_target_index_build_settings(
            "tfidf", tfidf_ngram_min=1, tfidf_ngram_max=1
        ),
    )
    src_matrix = target_index.vectorizer.transform(source_names)

    baseline = _run_backend(
        target_index,
        src_matrix,
        source_ids,
        true_target_by_source,
        top_k=top_k,
        backend_options=None,
    )
    filtered = _run_backend(
        target_index,
        src_matrix,
        source_ids,
        true_target_by_source,
        top_k=top_k,
        backend_options={"prefix_filter": True},
    )

    print(
        f"\n[prefix filter] n_rows={n_rows}: "
        f"baseline score={baseline['score_seconds']:.4f}s recall@{top_k}="
        f"{baseline['recall_at_k']:.4f} | "
        f"prefix_filter build={filtered['build_seconds']:.4f}s "
        f"score={filtered['score_seconds']:.4f}s recall@{top_k}="
        f"{filtered['recall_at_k']:.4f}"
    )

    # The filter is exact for the cosine scored, so at a threshold of zero it
    # returns every candidate the exhaustive scan ranks with a nonzero score;
    # recall at k cannot differ.
    assert filtered["recall_at_k"] == baseline["recall_at_k"]
