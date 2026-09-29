from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import company_vectorize
import numpy as np
import polars as pl
import pytest
from company_vectorize import resolve_target_index_build_settings
from company_vectorize.clustering_contract import TargetClusteringIndex
from company_vectorize.lsh_similarity import (
    DEFAULT_LSH_NUM_BANDS,
    DEFAULT_LSH_NUM_PERM,
    MinHashLshParams,
    build_target_lsh_index,
    score_source_with_lsh_backend,
)
from company_vectorize.sparse_similarity import (
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy
from scipy.sparse import csr_matrix

# The subprocess below needs this package importable. Taken from the package
# this test already imports, not by climbing to the repository root and back
# down: a package's tests stay inside the package, and `packages/` may not
# depend on `src/workspace` (see `src/workspace/README.md`'s Boundaries), so
# there is no shared anchor to reach for here and none is wanted.
VECTORIZE_SRC = Path(company_vectorize.__file__).resolve().parents[1]


def _build_target_index(names: list[str], ids: list[str] | None = None):
    strategy = TfidfClusteringStrategy()
    target = pl.DataFrame(
        {
            "system_uri": ids or [f"t{i + 1}" for i in range(len(names))],
            "name": names,
        }
    )
    return strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=resolve_target_index_build_settings(
            "tfidf", tfidf_ngram_min=1, tfidf_ngram_max=1
        ),
    )


def _dense_target_index() -> TargetClusteringIndex:
    return TargetClusteringIndex(
        target_ids=["t1", "t2"],
        vectorizer=None,
        target_matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_resolve_lsh_params_rejects_uneven_bands():
    target_index = _build_target_index(["acme limited", "beta partners"])
    with pytest.raises(ValueError, match="evenly divisible"):
        build_target_lsh_index(
            target_index, backend_options={"num_perm": 10, "num_bands": 3}
        )


def test_resolve_lsh_params_rejects_non_positive_values():
    target_index = _build_target_index(["acme limited", "beta partners"])
    with pytest.raises(ValueError, match="num_perm must be greater than zero"):
        build_target_lsh_index(target_index, backend_options={"num_perm": 0})
    with pytest.raises(ValueError, match="num_bands must be greater than zero"):
        build_target_lsh_index(
            target_index, backend_options={"num_perm": 8, "num_bands": 0}
        )


def test_lsh_rejects_dense_target_matrix():
    target_index = _dense_target_index()
    with pytest.raises(ValueError, match="requires a sparse target matrix"):
        build_target_lsh_index(target_index)


def test_lsh_rejects_dense_source_matrix():
    target_index = _build_target_index(["acme limited", "beta partners"])
    lsh_index = build_target_lsh_index(target_index)
    with pytest.raises(ValueError, match="requires a sparse source matrix"):
        score_source_with_lsh_backend(
            ["s1"],
            np.array([[1.0, 0.0]], dtype=np.float32),
            target_index=target_index,
            lsh_index=lsh_index,
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
        )


def test_build_target_lsh_index_empty_target_returns_none():
    target_index = TargetClusteringIndex(
        target_ids=[], vectorizer=None, target_matrix=csr_matrix((0, 0))
    )
    assert build_target_lsh_index(target_index) is None


def test_score_source_with_lsh_backend_returns_empty_without_index():
    target_index = _build_target_index(["acme limited"])
    rows = score_source_with_lsh_backend(
        ["s1"],
        target_index.vectorizer.transform(["acme limited"]),
        target_index=target_index,
        lsh_index=None,
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )
    assert rows == []


def test_target_lsh_index_has_one_bucket_dict_per_band():
    target_index = _build_target_index(["acme limited", "beta partners", "acme corp"])
    lsh_index = build_target_lsh_index(
        target_index, backend_options={"num_perm": 12, "num_bands": 4}
    )
    assert lsh_index is not None
    assert lsh_index.params == MinHashLshParams(num_perm=12, num_bands=4, seed=42)
    assert len(lsh_index.buckets) == 4
    assert lsh_index.target_signatures.shape == (3, 12)


def test_lsh_defaults_used_when_no_backend_options_given():
    target_index = _build_target_index(["acme limited"])
    lsh_index = build_target_lsh_index(target_index)
    assert lsh_index is not None
    assert lsh_index.params.num_perm == DEFAULT_LSH_NUM_PERM
    assert lsh_index.params.num_bands == DEFAULT_LSH_NUM_BANDS


def test_exact_duplicate_source_row_is_always_a_top_candidate():
    # t1 and t2 have identical vocabulary-column sets (same text), so their
    # minhash signatures are byte-identical and they collide in every band --
    # this holds for any seed/num_perm/num_bands, not just the ones tried
    # here, unlike a "near-duplicate" scenario whose recall genuinely depends
    # on the LSH parameters.
    target_index = _build_target_index(
        ["acme limited", "acme limited", "zebra unrelated widgets inc"]
    )
    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="lsh",
        top_k=3,
        max_candidates_per_source=None,
        backend_options={"num_perm": 16, "num_bands": 4},
    )
    assert backend_index is not None
    assert backend_index.backend == "lsh"
    assert backend_index.lsh is not None

    rows = score_source_with_backend(
        ["s1"],
        target_index.vectorizer.transform(["acme limited"]),
        target_index=target_index,
        top_k=3,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="lsh",
        backend_index=backend_index,
    )

    matched_target_ids = {row["target_id"] for row in rows}
    assert {"t1", "t2"} <= matched_target_ids
    for row in rows:
        if row["target_id"] in ("t1", "t2"):
            assert row["similarity"] == pytest.approx(1.0)
    top = min(rows, key=lambda row: row["rank"])
    assert top["target_id"] in ("t1", "t2")
    assert top["similarity"] == pytest.approx(1.0)


def test_lsh_backend_requires_build_backend_index_first():
    target_index = _build_target_index(["acme limited"])
    with pytest.raises(ValueError, match="requires build_backend_index"):
        score_source_with_backend(
            ["s1"],
            target_index.vectorizer.transform(["acme limited"]),
            target_index=target_index,
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
            nn_index=None,
            backend="lsh",
            backend_index=None,
        )


def test_min_similarity_filters_out_low_agreement_candidates():
    target_index = _build_target_index(["acme limited", "acme limited"])
    lsh_index = build_target_lsh_index(
        target_index, backend_options={"num_perm": 16, "num_bands": 4}
    )
    rows = score_source_with_lsh_backend(
        ["s1"],
        target_index.vectorizer.transform(["acme limited"]),
        target_index=target_index,
        lsh_index=lsh_index,
        top_k=5,
        min_similarity=1.5,
        max_candidates_per_source=None,
    )
    assert rows == []


def test_max_candidates_per_source_caps_rows_returned():
    names = ["acme limited"] * 5
    target_index = _build_target_index(names)
    lsh_index = build_target_lsh_index(
        target_index, backend_options={"num_perm": 16, "num_bands": 4}
    )
    rows = score_source_with_lsh_backend(
        ["s1"],
        target_index.vectorizer.transform(["acme limited"]),
        target_index=target_index,
        lsh_index=lsh_index,
        top_k=5,
        min_similarity=0.0,
        max_candidates_per_source=2,
    )
    assert len(rows) == 2


def test_source_row_with_no_terms_gets_no_candidates():
    # An all-zero source row (empty text vectorized through a fitted TF-IDF
    # vectorizer has no nonzero columns) exercises this module's own
    # defensive path for a row with no vocabulary terms at all: its minhash
    # signature is the all-sentinel placeholder (see
    # `_minhash_signature_matrix`'s docstring), which by construction never
    # matches a real bucket key, so the row is silently skipped rather than
    # erroring.
    target_index = _build_target_index(["acme limited", "beta partners"])
    lsh_index = build_target_lsh_index(
        target_index, backend_options={"num_perm": 16, "num_bands": 4}
    )
    rows = score_source_with_lsh_backend(
        ["s1"],
        target_index.vectorizer.transform([""]),
        target_index=target_index,
        lsh_index=lsh_index,
        top_k=5,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )
    assert rows == []


def test_signatures_deterministic_across_independent_builds():
    target_index = _build_target_index(
        ["acme limited", "beta partners", "gamma holdings", "delta group"]
    )
    first = build_target_lsh_index(
        target_index, backend_options={"num_perm": 20, "num_bands": 5, "seed": 7}
    )
    second = build_target_lsh_index(
        target_index, backend_options={"num_perm": 20, "num_bands": 5, "seed": 7}
    )
    assert first is not None and second is not None
    assert np.array_equal(first.target_signatures, second.target_signatures)
    assert first.buckets[0].keys() == second.buckets[0].keys()
    for key, rows in first.buckets[0].items():
        assert np.array_equal(rows, second.buckets[0][key])


@pytest.mark.integration
def test_signatures_are_identical_across_processes(tmp_path):
    """The determinism guarantee is literally cross-process, so assert it that way.

    Runs the same minhash-signature computation in a fresh subprocess (with
    the worktree's own package source on `sys.path`, matching
    `scripts/run_blocking.py`'s pattern for direct execution against the
    shared, possibly differently-synced `.venv`) and compares its output to
    the in-process result. The target matrix and signature output round-trip
    through files rather than inline source, so this stays a plain
    determinism check rather than a `subprocess`-`eval` code-injection risk.
    """
    from scipy.sparse import save_npz

    target_index = _build_target_index(["acme limited", "beta partners"])
    in_process = build_target_lsh_index(
        target_index, backend_options={"num_perm": 12, "num_bands": 3, "seed": 99}
    )
    assert in_process is not None

    matrix_path = tmp_path / "target_matrix.npz"
    signatures_path = tmp_path / "signatures.npy"
    save_npz(matrix_path, target_index.target_matrix.tocsr())

    script_path = tmp_path / "compute_signatures.py"
    script_path.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(VECTORIZE_SRC)!r})\n"
        "import numpy as np\n"
        "from scipy.sparse import load_npz\n"
        "from company_vectorize.lsh_similarity import (\n"
        "    _hash_coefficients,\n"
        "    _minhash_signature_matrix,\n"
        ")\n"
        f"matrix = load_npz({str(matrix_path)!r})\n"
        "a, b = _hash_coefficients(12, 99)\n"
        "signatures = _minhash_signature_matrix(matrix, a=a, b=b)\n"
        f"np.save({str(signatures_path)!r}, signatures)\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, check=True
    )
    subprocess_signatures = np.load(signatures_path)
    assert np.array_equal(subprocess_signatures, in_process.target_signatures)
