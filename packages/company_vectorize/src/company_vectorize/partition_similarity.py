from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import issparse
from sklearn.cluster import HDBSCAN, KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.neighbors import NearestNeighbors

from .clustering_contract import TargetClusteringIndex
from .ranking import append_ranked_match_block

# "kmeans"/"hdbscan" are candidate-generation backends, not a clustering
# replacement: they partition the
# *target* index once, route each source row to one partition, and score only
# within it -- the same job the `lsh` backend's hash buckets do, via centroids/density
# instead of hashes. There is no default partition-size capping. An oversized
# partition (a K-means cluster that swallowed half the targets, or an HDBSCAN
# noise blob) is a real, reportable strategy outcome -- see
# `partition_cluster_shape_frame()`, which feeds `compute_cluster_shape_metrics`
# unchanged so that outcome shows up in comparison reports rather than being
# engineered away.
#
# HDBSCAN-specific limitation, stated plainly rather than hidden: sklearn's
# `HDBSCAN` has no `.predict()` for unseen rows (it's a transductive, density-based
# fit) and no sparse-input support at real vocabulary sizes. Source-row routing
# here approximates assignment via nearest fitted-cluster centroid (`_CentroidRouter`,
# in the spirit of the standalone `hdbscan` package's `approximate_predict()`) --
# NOT true density membership. Target rows sklearn labels noise (`-1`) get their
# own partition for size-reporting purposes but are never a routing target, so
# they are structurally unreachable by this backend -- a genuine, measured recall
# cost of choosing HDBSCAN as a blocking key, not a bug.

# Measured on `gleif -> ie` (0.8M target rows, tfidf), of the pairs no name
# equality resolves: 300 clusters found as many as 900 or 2,400 and more than
# 50, and the fit's time grows with the count while scoring stays seconds at
# every count, so a few hundred is cheaper than the square root of the target
# and loses nothing. One target so far; docs/findings/candidate-generation.md
# has the runs.
DEFAULT_KMEANS_CLUSTERS = 300
KMEANS_CLUSTERS_OPTION = "kmeans_clusters"

# The kmeans fit is described by the partition it makes (cluster count, start,
# seed), by how many target rows each of its steps reads, and by when it stops.
# Which scikit-learn estimator runs follows from those, rather than being a
# setting of its own.
#
# A row budget is `all` or a number of rows. `kmeans_start_rows` is the rows
# the start picks its centroids from: a sample of at least ten per cluster
# when a number, the whole target when `all` or when the target is no larger.
# `kmeans_fit_rows` is the rows each iteration reads: `all` is the full fit,
# a number smaller than the target is scikit-learn's mini-batch fit, reading
# batches of that many rows.
ALL_ROWS = "all"
KMEANS_START_ROWS_OPTION = "kmeans_start_rows"
KMEANS_FIT_ROWS_OPTION = "kmeans_fit_rows"
DEFAULT_KMEANS_START_ROWS = "50000"
DEFAULT_KMEANS_FIT_ROWS = ALL_ROWS

# How the start picks its centroids. `spread` is scikit-learn's k-means++,
# picking one at a time, far from those already picked, measuring every start
# row at each pick, so its cost grows with start rows times clusters: measured
# on `ie` (0.8M rows, tfidf), 900 clusters spent about 29 of a 36-minute fit
# there when it read every row. `random` takes that many start rows as they
# come, at no cost and with no spreading. On `ie` the spread start over every
# row, over a 50,000-row sample and the random start found the same pairs to
# within the spread three seeds of one start give, so the default is the
# spread start over a sample, as tight a fit as over every row at under half
# the time.
KMEANS_START_SPREAD = "spread"
KMEANS_START_RANDOM = "random"
KMEANS_START_CHOICES: tuple[str, ...] = (KMEANS_START_SPREAD, KMEANS_START_RANDOM)
DEFAULT_KMEANS_START = KMEANS_START_SPREAD
KMEANS_START_OPTION = "kmeans_start"
_SKLEARN_START: dict[str, str] = {
    KMEANS_START_SPREAD: "k-means++",
    KMEANS_START_RANDOM: "random",
}
_KMEANS_START_ROWS_PER_CLUSTER = 10

# Independent fits, each from its own start, of which the tightest is kept.
# One: the partitions only route a source to a slice of the target, so
# restarts buy tidier clusters at a multiple of the fit for nothing a run
# measures. Restart `i` starts from seed `seed + i`.
DEFAULT_KMEANS_RESTARTS = 1
KMEANS_RESTARTS_OPTION = "kmeans_restarts"

# The seed, read from the options as `seed`, the key the lsh backend's seed
# uses. 42 is the value the fit had when it was fixed in the package, so a run
# that names none fits as runs made before it did.
DEFAULT_KMEANS_SEED = 42
KMEANS_SEED_OPTION = "seed"

# When a fit stops: after `max_passes` passes over the target, or once the
# centroids move less than `stop_tolerance`, or, for a mini-batch fit, after
# `stall_batches` batches in a row that did not improve it. Measured on `ie`
# (0.8M rows, tfidf, 50 clusters), inertia above its value at convergence
# (iteration 99): 0.48% at iteration 10, 0.036% at 20, 0.008% at 50. The fit
# is flat from 20, which is all a routing partition needs. The tolerance and
# stall count are scikit-learn's defaults for the full and mini-batch fits.
MAX_PASSES_OPTION = "max_passes"
STOP_TOLERANCE_OPTION = "stop_tolerance"
STALL_BATCHES_OPTION = "stall_batches"
DEFAULT_KMEANS_MAX_PASSES = 25
DEFAULT_KMEANS_STOP_TOLERANCE = 1e-4
DEFAULT_KMEANS_STALL_BATCHES = 10

# A mini-batch fit moves the centroid of a partition smaller than this fraction
# of the largest one to a fresh row. scikit-learn's default.
KMEANS_BATCH_RESEED_BELOW_OPTION = "kmeans_batch_reseed_below"
DEFAULT_KMEANS_BATCH_RESEED_BELOW = 0.01

DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_CLUSTER_SIZE_OPTION = "min_cluster_size"
# The neighbours a row needs to be a core point; unset is the minimum cluster
# size, as in scikit-learn, resolved here so the fit never rests on a library
# default.
HDBSCAN_MIN_SAMPLES_OPTION = "hdbscan_min_samples"
DEFAULT_HDBSCAN_MIN_SAMPLES: int | None = None
# How clusters are chosen from the density tree: `eom` keeps the most stable
# ones, `leaf` the finest. scikit-learn's default is `eom`.
HDBSCAN_SELECTION_OPTION = "hdbscan_selection"
HDBSCAN_SELECTION_CHOICES: tuple[str, ...] = ("eom", "leaf")
DEFAULT_HDBSCAN_SELECTION = "eom"

NOISE_LABEL = -1


class _CentroidRouter:
    """Routes rows to the nearest HDBSCAN cluster centroid.

    Substitutes for sklearn `HDBSCAN`'s missing `.predict()`. Centroid row `i`
    is assumed to correspond to cluster label `i` (sklearn's
    `store_centers="centroid"` ordering); noise (`NOISE_LABEL`) is never a
    routing target by construction -- it simply isn't one of `centroids`.
    """

    def __init__(self, centroids: np.ndarray) -> None:
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
        self._nn.fit(centroids)

    def predict(self, matrix: Any) -> np.ndarray:
        _, indices = self._nn.kneighbors(matrix, return_distance=True)
        return indices[:, 0]


@dataclass(frozen=True)
class TargetPartitionIndex:
    """Fitted partition state for the `"kmeans"`/`"hdbscan"` similarity backends.

    Built by `build_target_partition_index()` and stored on
    `TargetSimilarityBackendIndex.partition`; not normally constructed directly.

    Attributes:
        backend: `"kmeans"` or `"hdbscan"`.
        labels: Partition label per target row, aligned by position with
            `TargetClusteringIndex.target_ids` (`NOISE_LABEL` marks HDBSCAN
            noise -- see module docstring).
        router: `.predict(matrix) -> label array`, used to route source rows
            to a partition. The fitted `KMeans` or `MiniBatchKMeans` itself
            for `"kmeans"`; a
            `_CentroidRouter` for `"hdbscan"`; `None` if HDBSCAN found no real
            clusters to route into (every target row was noise) -- routing
            then produces zero candidates for every source row, an honest
            outcome rather than a silent fallback.
        partition_neighbor_index: partition label -> fitted `NearestNeighbors`
            over just that partition's target rows.
        partition_local_to_target_index: partition label -> array mapping a
            `partition_neighbor_index` result position back to a row index in
            `TargetClusteringIndex.target_matrix`/`target_ids`.
    """

    backend: str
    labels: np.ndarray
    router: Any | None
    partition_neighbor_index: dict[int, NearestNeighbors]
    partition_local_to_target_index: dict[int, np.ndarray]


def _resolve_choice(
    options: dict[str, object], key: str, default: str, choices: tuple[str, ...]
) -> str:
    choice = str(options.get(key, default)).strip()
    if choice not in choices:
        raise ValueError(f"{key} must be one of: {', '.join(choices)}; got {choice!r}")
    return choice


def _resolve_row_budget(
    options: dict[str, object], key: str, default: str
) -> int | None:
    """A row budget as a number of rows, or `None` for every row."""
    raw = options.get(key, default)
    text = str(raw).strip().lower()
    if text == ALL_ROWS:
        return None
    try:
        rows = int(text)
    except ValueError:
        rows = 0
    if rows <= 0:
        raise ValueError(
            f"{key} must be '{ALL_ROWS}' or a positive number of rows; got {raw!r}"
        )
    return rows


def _resolve_kmeans_start(
    options: dict[str, object],
    *,
    target_matrix: Any,
    n_clusters: int,
    start_rows: int | None,
    seed: int,
) -> tuple[Any, int | None]:
    """The `init` for the start the options name, and the rows it reads.

    The rows are for a mini-batch fit, whose estimator samples them itself.
    Over every row, the start is scikit-learn's own, named, with every row as
    a mini-batch fit's `init_size`. Over a sample, the centroids are picked
    here from the sample and passed in, so both estimators start from the same
    ones. A target no larger than the sample is started over all of it, since
    sampling it would change nothing.
    """
    start = _resolve_choice(
        options, KMEANS_START_OPTION, DEFAULT_KMEANS_START, KMEANS_START_CHOICES
    )
    n_targets = target_matrix.shape[0]
    sample_rows = (
        None
        if start_rows is None
        else max(start_rows, _KMEANS_START_ROWS_PER_CLUSTER * n_clusters)
    )
    if sample_rows is None or n_targets <= sample_rows:
        return _SKLEARN_START[start], n_targets
    rows = np.sort(
        np.random.default_rng(seed).choice(n_targets, size=sample_rows, replace=False)
    )
    if start == KMEANS_START_SPREAD:
        centers, _ = kmeans_plusplus(
            target_matrix[rows], n_clusters=n_clusters, random_state=seed
        )
        return centers, None
    picked = np.sort(
        np.random.default_rng(seed).choice(rows, size=n_clusters, replace=False)
    )
    chosen = target_matrix[picked]
    return (chosen.toarray() if issparse(chosen) else np.asarray(chosen)), None


def _resolve_int_option(options: dict[str, object], key: str, default: int) -> int:
    raw = options.get(key, default)
    return int(raw) if isinstance(raw, (int, float, str)) else default


def _resolve_optional_int_option(
    options: dict[str, object], key: str, default: int | None
) -> int | None:
    raw = options.get(key, default)
    return int(raw) if isinstance(raw, (int, float, str)) else default


def _resolve_float_option(
    options: dict[str, object], key: str, default: float
) -> float:
    raw = options.get(key, default)
    return float(raw) if isinstance(raw, (int, float, str)) else default


# A fit over a large target runs for minutes in silence, so past this size a
# full fit prints each iteration; it changes nothing fitted. A mini-batch fit
# stays quiet, since it would print every batch.
_KMEANS_VERBOSE_MIN_ROWS = 100_000


def _fit_kmeans(
    target_matrix: Any, options: dict[str, object], *, n_targets: int
) -> KMeans | MiniBatchKMeans:
    """The tightest of the fits the options ask for, by inertia."""
    n_clusters = _resolve_int_option(
        options, KMEANS_CLUSTERS_OPTION, DEFAULT_KMEANS_CLUSTERS
    )
    n_clusters = max(1, min(n_clusters, n_targets))
    seed = _resolve_int_option(options, KMEANS_SEED_OPTION, DEFAULT_KMEANS_SEED)
    restarts = max(
        1,
        _resolve_int_option(options, KMEANS_RESTARTS_OPTION, DEFAULT_KMEANS_RESTARTS),
    )
    start_rows = _resolve_row_budget(
        options, KMEANS_START_ROWS_OPTION, DEFAULT_KMEANS_START_ROWS
    )
    fit_rows = _resolve_row_budget(
        options, KMEANS_FIT_ROWS_OPTION, DEFAULT_KMEANS_FIT_ROWS
    )
    max_passes = _resolve_int_option(
        options, MAX_PASSES_OPTION, DEFAULT_KMEANS_MAX_PASSES
    )
    stop_tolerance = _resolve_float_option(
        options, STOP_TOLERANCE_OPTION, DEFAULT_KMEANS_STOP_TOLERANCE
    )
    batch_rows = n_targets if fit_rows is None else min(fit_rows, n_targets)
    mini_batch = batch_rows < n_targets

    def _fit_once(restart_seed: int) -> KMeans | MiniBatchKMeans:
        init, init_rows = _resolve_kmeans_start(
            options,
            target_matrix=target_matrix,
            n_clusters=n_clusters,
            start_rows=start_rows,
            seed=restart_seed,
        )
        model: KMeans | MiniBatchKMeans
        if mini_batch:
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                init=init,
                init_size=init_rows,
                batch_size=batch_rows,
                n_init=1,
                max_iter=max_passes,
                tol=stop_tolerance,
                max_no_improvement=_resolve_int_option(
                    options, STALL_BATCHES_OPTION, DEFAULT_KMEANS_STALL_BATCHES
                ),
                reassignment_ratio=_resolve_float_option(
                    options,
                    KMEANS_BATCH_RESEED_BELOW_OPTION,
                    DEFAULT_KMEANS_BATCH_RESEED_BELOW,
                ),
                random_state=restart_seed,
            )
        else:
            model = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=1,
                max_iter=max_passes,
                tol=stop_tolerance,
                random_state=restart_seed,
                verbose=int(n_targets >= _KMEANS_VERBOSE_MIN_ROWS),
            )
        return model.fit(target_matrix)

    # Only the tightest fit so far is held: a centroid matrix is clusters by
    # vocabulary, dense.
    best = _fit_once(seed)
    for restart in range(1, restarts):
        model = _fit_once(seed + restart)
        if model.inertia_ < best.inertia_:
            best = model
    return best


def build_target_partition_index(
    target_index: TargetClusteringIndex,
    *,
    backend: str,
    backend_options: dict[str, object] | None = None,
) -> TargetPartitionIndex | None:
    n_targets = len(target_index.target_ids)
    if n_targets == 0:
        return None
    options = backend_options or {}

    if backend == "kmeans":
        kmeans = _fit_kmeans(target_index.target_matrix, options, n_targets=n_targets)
        labels = kmeans.labels_
        router: Any | None = kmeans
    elif backend == "hdbscan":
        if issparse(target_index.target_matrix):
            raise ValueError(
                "hdbscan backend requires a dense target matrix (for example the "
                "'sbert' representation); sparse representations like "
                "tfidf/wordpiece/sentencepiece are not supported -- densifying "
                "them would be infeasible at real vocabulary sizes. Use the "
                "kmeans backend instead, which supports both."
            )
        min_cluster_size = _resolve_int_option(
            options, HDBSCAN_MIN_CLUSTER_SIZE_OPTION, DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
        )
        min_cluster_size = max(2, min(min_cluster_size, n_targets))
        min_samples = _resolve_optional_int_option(
            options, HDBSCAN_MIN_SAMPLES_OPTION, DEFAULT_HDBSCAN_MIN_SAMPLES
        )
        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size if min_samples is None else min_samples,
            cluster_selection_method=_resolve_choice(
                options,
                HDBSCAN_SELECTION_OPTION,
                DEFAULT_HDBSCAN_SELECTION,
                HDBSCAN_SELECTION_CHOICES,
            ),
            metric="cosine",
            store_centers="centroid",
            copy=False,
        )
        labels = hdbscan.fit_predict(target_index.target_matrix)
        router = (
            _CentroidRouter(hdbscan.centroids_) if hdbscan.centroids_.shape[0] else None
        )
    else:
        raise ValueError(f"Unsupported partition backend: {backend}")

    partition_neighbor_index: dict[int, NearestNeighbors] = {}
    partition_local_to_target_index: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        member_indices = np.flatnonzero(labels == label)
        partition_local_to_target_index[int(label)] = member_indices
        # One job, deliberately. sklearn chunks a brute search to a memory
        # budget and computes several chunks at once when given more, each
        # allocating its own distance block over the partition: `n_jobs=-1`
        # died on `gb`, whose largest kmeans partition holds 2,037,794 of
        # its 5.7M targets, at a gibibyte per index array. Parallelism here
        # needs `sklearn.config_context(working_memory=...)` to bound the
        # block first, measured on a real run.
        nn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=len(member_indices)
        )
        nn.fit(target_index.target_matrix[member_indices])
        partition_neighbor_index[int(label)] = nn

    return TargetPartitionIndex(
        backend=backend,
        labels=labels,
        router=router,
        partition_neighbor_index=partition_neighbor_index,
        partition_local_to_target_index=partition_local_to_target_index,
    )


def score_source_with_partition_backend(
    src_ids: list[str],
    src_matrix: Any,
    *,
    target_index: TargetClusteringIndex,
    partition_index: TargetPartitionIndex | None,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
) -> list[dict[str, object]]:
    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    if partition_index is None or partition_index.router is None or max_per_source <= 0:
        return []

    router_labels = np.asarray(partition_index.router.predict(src_matrix))

    # One neighbour query per partition, over every source row routed to it:
    # a row's candidates depend only on that row and its partition, so rows
    # sharing a partition are searched as one matrix product rather than one
    # call each, which is what a query per row cost at a large target.
    # One block for every source row, `-inf` where a row's partition gave it
    # fewer than `max_per_source` candidates: a padded entry never clears
    # `min_similarity`, so it is dropped where a shorter row simply ended.
    block_scores = np.full((len(src_ids), max_per_source), -np.inf, dtype=np.float64)
    block_indices = np.zeros((len(src_ids), max_per_source), dtype=np.int64)
    for label in np.unique(router_labels):
        nn = partition_index.partition_neighbor_index.get(int(label))
        local_to_target = partition_index.partition_local_to_target_index.get(
            int(label)
        )
        if nn is None or local_to_target is None or local_to_target.size == 0:
            continue
        routed = np.flatnonzero(router_labels == label)
        distances, local_indices = nn.kneighbors(
            src_matrix[routed],
            n_neighbors=min(max_per_source, local_to_target.size),
            return_distance=True,
        )
        width = local_indices.shape[1]
        block_indices[routed, :width] = local_to_target[local_indices]
        block_scores[routed, :width] = 1.0 - np.asarray(distances, dtype=np.float64)

    # Emitted in source order, as a query per row emitted them.
    rows: list[dict[str, object]] = []
    append_ranked_match_block(
        rows,
        src_ids=src_ids,
        target_ids=target_index.target_ids,
        candidate_indices=block_indices,
        candidate_scores=block_scores,
        min_similarity=min_similarity,
        max_per_source=max_per_source,
    )
    return rows


def partition_cluster_shape_frame(
    partition_index: TargetPartitionIndex, target_ids: list[str]
) -> pl.DataFrame:
    """Project a `TargetPartitionIndex` into the shape cluster-shape metrics expect.

    Produces the `{cluster_id, node_id}` frame `compute_cluster_shape_metrics()` takes, so
    partition size, including any oversized partition or HDBSCAN's noise bucket, reports
    through the same metric connected-components clusters already use. No separate
    metric family for partition-based backends.
    """
    labels = partition_index.labels
    return pl.DataFrame(
        {
            "cluster_id": [
                f"{partition_index.backend}-{int(label)}" for label in labels
            ],
            "node_id": target_ids,
        }
    )
