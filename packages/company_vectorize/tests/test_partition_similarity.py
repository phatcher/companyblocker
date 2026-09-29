import numpy as np
import pytest
from company_vectorize import (
    TargetClusteringIndex,
    build_target_partition_index,
    compute_cluster_shape_metrics,
    partition_cluster_shape_frame,
)
from company_vectorize.sparse_similarity import (
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from scipy.sparse import csr_matrix


def _dense_target_index() -> TargetClusteringIndex:
    # Two obvious groups, well separated on the unit circle: (t1,t2) near
    # (1,0), (t3,t4) near (0,1).
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    return TargetClusteringIndex(
        target_ids=["t1", "t2", "t3", "t4"], vectorizer=None, target_matrix=matrix
    )


def _sparse_target_index() -> TargetClusteringIndex:
    dense = _dense_target_index()
    return TargetClusteringIndex(
        target_ids=dense.target_ids,
        vectorizer=None,
        target_matrix=csr_matrix(dense.target_matrix),
    )


def test_kmeans_partitions_sparse_and_dense_and_routes_correctly():
    for target_index in (_sparse_target_index(), _dense_target_index()):
        partition_index = build_target_partition_index(
            target_index, backend="kmeans", backend_options={"kmeans_clusters": 2}
        )
        assert partition_index is not None
        assert len(partition_index.labels) == 4
        # t1/t2 must land in the same partition, t3/t4 in the same partition,
        # and the two partitions must differ.
        assert partition_index.labels[0] == partition_index.labels[1]
        assert partition_index.labels[2] == partition_index.labels[3]
        assert partition_index.labels[0] != partition_index.labels[2]

        # A source row near (1, 0) should route to t1/t2's partition and match t1 best.
        source = target_index.target_matrix[0:1]
        router_label = int(partition_index.router.predict(source)[0])
        assert router_label == int(partition_index.labels[0])


def test_kmeans_backend_scores_source_chunk_end_to_end():
    target_index = _dense_target_index()
    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="kmeans",
        top_k=2,
        backend_options={"kmeans_clusters": 2},
    )
    rows = score_source_with_backend(
        ["s1"],
        target_index.target_matrix[0:1],
        target_index=target_index,
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="kmeans",
        backend_index=backend_index,
    )
    assert rows
    top = min(rows, key=lambda row: row["rank"])
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t1"
    # Only t1/t2 (source's own partition) are reachable, never t3/t4.
    assert {row["target_id"] for row in rows} <= {"t1", "t2"}


def test_kmeans_backend_scores_a_chunk_as_it_scores_each_row_alone():
    """Rows sharing a partition are searched together, so a chunk must give
    each row the candidates, scores and ranks it gets when scored alone, in
    source order, sparse and dense alike."""
    rng = np.random.default_rng(7)
    targets = np.abs(rng.normal(size=(60, 8))).astype(np.float32)
    sources = np.abs(rng.normal(size=(25, 8))).astype(np.float32)
    source_ids = [f"s{index}" for index in range(sources.shape[0])]

    for as_matrix in (np.asarray, csr_matrix):
        target_index = TargetClusteringIndex(
            target_ids=[f"t{index}" for index in range(targets.shape[0])],
            vectorizer=None,
            target_matrix=as_matrix(targets),
        )
        backend_index = build_target_similarity_backend_index(
            target_index,
            backend="kmeans",
            top_k=3,
            backend_options={"kmeans_clusters": 4},
        )

        def score(ids, matrix, target_index=target_index, backend_index=backend_index):
            return score_source_with_backend(
                ids,
                matrix,
                target_index=target_index,
                top_k=3,
                min_similarity=0.2,
                max_candidates_per_source=None,
                nn_index=None,
                backend="kmeans",
                backend_index=backend_index,
            )

        source_matrix = as_matrix(sources)
        together = score(source_ids, source_matrix)
        alone = [
            row
            for index, source_id in enumerate(source_ids)
            for row in score([source_id], source_matrix[index : index + 1])
        ]

        assert together
        assert [(r["source_id"], r["target_id"], r["rank"]) for r in together] == [
            (r["source_id"], r["target_id"], r["rank"]) for r in alone
        ]
        assert [r["similarity"] for r in together] == pytest.approx(
            [r["similarity"] for r in alone]
        )


def _random_sparse_target(rows: int, seed: int) -> TargetClusteringIndex:
    rng = np.random.default_rng(seed)
    return TargetClusteringIndex(
        target_ids=[f"t{index}" for index in range(rows)],
        vectorizer=None,
        target_matrix=csr_matrix(np.abs(rng.normal(size=(rows, 6))).astype(np.float32)),
    )


def _record_estimator(monkeypatch, name: str) -> list[dict[str, object]]:
    """Every keyword set the partition backend builds `name` with."""
    from company_vectorize import partition_similarity

    seen: list[dict[str, object]] = []
    real = getattr(partition_similarity, name)

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(partition_similarity, name, recording)
    return seen


def test_kmeans_start_picks_from_the_start_rows_and_refuses_an_unknown_one(
    monkeypatch,
):
    """Over every row the start is the library's own, `k-means++` for spread
    and `random` for random; over a sample the fit is handed centroids picked
    from it. A target no larger than the sample is started over all of it,
    which is what the default sample gives a small target."""
    target_index = _random_sparse_target(80, seed=3)
    dense_rows = target_index.target_matrix.toarray()
    seen = _record_estimator(monkeypatch, "KMeans")

    def fit(**options: object) -> None:
        partition_index = build_target_partition_index(
            target_index,
            backend="kmeans",
            backend_options={"kmeans_clusters": 4, **options},
        )
        assert partition_index is not None
        assert len(partition_index.labels) == 80

    fit()
    fit(kmeans_start="spread", kmeans_start_rows="all")
    fit(kmeans_start="random", kmeans_start_rows="all")
    # At least ten rows per cluster, so 40 is the sample 4 clusters take.
    fit(kmeans_start="spread", kmeans_start_rows="40")
    fit(kmeans_start="random", kmeans_start_rows=40)

    assert [kwargs["init"] for kwargs in seen[:3]] == [
        "k-means++",
        "k-means++",
        "random",
    ]
    assert isinstance(seen[3]["init"], np.ndarray)
    assert seen[3]["init"].shape == (4, 6)
    picked = seen[4]["init"]
    assert isinstance(picked, np.ndarray) and picked.shape == (4, 6)
    assert all((dense_rows == row).all(axis=1).any() for row in picked)

    with pytest.raises(ValueError, match="kmeans_start must be one of"):
        fit(kmeans_start="k-means++")
    with pytest.raises(ValueError, match="kmeans_start_rows must be 'all'"):
        fit(kmeans_start_rows="some")


def test_kmeans_seed_option_seeds_the_fit_and_defaults_to_the_fixed_seed(
    monkeypatch,
):
    """A fit that names no seed is seeded 42, the value fixed in the package
    before the seed was an option, so it fits as those runs did; a named seed
    reaches the fit and the sample its start is drawn from."""
    target_index = _random_sparse_target(80, seed=5)
    seen = _record_estimator(monkeypatch, "KMeans")

    def fit(**options: object) -> np.ndarray:
        partition_index = build_target_partition_index(
            target_index,
            backend="kmeans",
            backend_options={
                "kmeans_clusters": 4,
                "kmeans_start_rows": "40",
                **options,
            },
        )
        assert partition_index is not None
        return partition_index.labels

    unnamed = fit()
    named_default = fit(seed=42)
    other = fit(seed=7)

    assert [kwargs["random_state"] for kwargs in seen] == [42, 42, 7]
    np.testing.assert_array_equal(unnamed, named_default)
    np.testing.assert_array_equal(seen[0]["init"], seen[1]["init"])
    assert not np.array_equal(seen[0]["init"], seen[2]["init"])
    assert len(other) == 80


def test_kmeans_fit_rows_below_the_target_fits_in_batches(monkeypatch):
    """A fit-row budget smaller than the target is the mini-batch fit, reading
    batches of that many rows and stopping by the stop settings; one no
    smaller is the full fit. Both route every target row."""
    target_index = _random_sparse_target(80, seed=9)
    batched = _record_estimator(monkeypatch, "MiniBatchKMeans")
    full = _record_estimator(monkeypatch, "KMeans")

    for fit_rows in ("20", "80"):
        partition_index = build_target_partition_index(
            target_index,
            backend="kmeans",
            backend_options={
                "kmeans_clusters": 4,
                "kmeans_fit_rows": fit_rows,
                "max_passes": 7,
                "stop_tolerance": 0.5,
                "stall_batches": 3,
                "kmeans_batch_reseed_below": 0.2,
            },
        )
        assert partition_index is not None
        assert len(partition_index.labels) == 80
        routed = partition_index.router.predict(target_index.target_matrix[:5])
        assert len(routed) == 5

    assert len(batched) == 1 and len(full) == 1
    assert batched[0]["batch_size"] == 20
    assert batched[0]["init_size"] == 80
    assert batched[0]["max_iter"] == 7
    assert batched[0]["tol"] == 0.5
    assert batched[0]["max_no_improvement"] == 3
    assert batched[0]["reassignment_ratio"] == 0.2
    assert full[0]["max_iter"] == 7
    assert full[0]["tol"] == 0.5


def test_kmeans_restarts_keep_the_tightest_fit(monkeypatch):
    """Each restart is its own fit from seed `seed + i`, and the partition is
    the one with the lowest inertia."""
    from company_vectorize import partition_similarity

    target_index = _random_sparse_target(80, seed=11)
    inertias: dict[int, float] = {}
    real = partition_similarity.KMeans

    def recording(*args, **kwargs):
        model = real(*args, **kwargs)
        real_fit = model.fit

        def fit(matrix):
            fitted = real_fit(matrix)
            inertias[kwargs["random_state"]] = fitted.inertia_
            return fitted

        model.fit = fit
        return model

    monkeypatch.setattr(partition_similarity, "KMeans", recording)
    partition_index = build_target_partition_index(
        target_index,
        backend="kmeans",
        backend_options={"kmeans_clusters": 4, "kmeans_restarts": 3, "seed": 5},
    )

    assert partition_index is not None
    assert sorted(inertias) == [5, 6, 7]
    assert partition_index.router.inertia_ == min(inertias.values())


def test_hdbscan_settings_reach_the_fit(monkeypatch):
    """An unset minimum sample count is the minimum cluster size, resolved
    here rather than left to the library; the selection method is passed as
    named, and an unknown one is refused."""
    target_index = _dense_target_index()
    seen = _record_estimator(monkeypatch, "HDBSCAN")

    build_target_partition_index(
        target_index, backend="hdbscan", backend_options={"min_cluster_size": 2}
    )
    build_target_partition_index(
        target_index,
        backend="hdbscan",
        backend_options={
            "min_cluster_size": 2,
            "hdbscan_min_samples": 1,
            "hdbscan_selection": "leaf",
        },
    )

    assert [(kw["min_samples"], kw["cluster_selection_method"]) for kw in seen] == [
        (2, "eom"),
        (1, "leaf"),
    ]
    with pytest.raises(ValueError, match="hdbscan_selection must be one of"):
        build_target_partition_index(
            target_index,
            backend="hdbscan",
            backend_options={"hdbscan_selection": "best"},
        )


def test_kmeans_backend_requires_build_backend_index_first():
    target_index = _dense_target_index()
    with pytest.raises(ValueError, match="requires build_backend_index"):
        score_source_with_backend(
            ["s1"],
            target_index.target_matrix[0:1],
            target_index=target_index,
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
            nn_index=None,
            backend="kmeans",
            backend_index=None,
        )


def test_hdbscan_rejects_sparse_target_matrix():
    target_index = _sparse_target_index()
    with pytest.raises(ValueError, match="requires a dense target matrix"):
        build_target_partition_index(target_index, backend="hdbscan")


def test_hdbscan_partitions_dense_target_and_routes():
    target_index = _dense_target_index()
    partition_index = build_target_partition_index(
        target_index, backend="hdbscan", backend_options={"min_cluster_size": 2}
    )
    assert partition_index is not None
    assert partition_index.labels[0] == partition_index.labels[1]
    assert partition_index.labels[2] == partition_index.labels[3]


def test_partition_backend_rejects_unknown_backend_name():
    target_index = _dense_target_index()
    with pytest.raises(ValueError, match="Unsupported partition backend"):
        build_target_partition_index(target_index, backend="unknown")


def test_partition_cluster_shape_frame_reports_oversized_partition_without_trimming():
    # A deliberately lopsided partitioning: one giant cluster, one singleton.
    # No capping should occur anywhere in this path -- the imbalance must be
    # visible in compute_cluster_shape_metrics, not silently corrected.
    target_index = TargetClusteringIndex(
        target_ids=[f"t{i}" for i in range(10)],
        vectorizer=None,
        target_matrix=np.zeros((10, 2), dtype=np.float32),
    )
    partition_index = build_target_partition_index(
        target_index, backend="kmeans", backend_options={"kmeans_clusters": 2}
    )
    assert partition_index is not None
    # Force a lopsided assignment directly to exercise the metrics path
    # deterministically, independent of what KMeans itself happens to converge to.
    partition_index.labels[:] = [0] * 9 + [1]

    frame = partition_cluster_shape_frame(partition_index, target_index.target_ids)
    assert frame.height == 10

    shape = compute_cluster_shape_metrics(clusters=frame)
    row = shape.row(0, named=True)
    assert row["total_clusters"] == 2
    assert row["max_cluster_size"] == 9
    assert row["singleton_clusters"] == 1
