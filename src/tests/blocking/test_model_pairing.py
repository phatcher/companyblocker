"""A known-answer test of the clustering assumptions.

`fixtures/model_pairing.json` is a checked-in ~10-rows-a-side population, one
source row per clustering edge case (the exact-join cascade's raw/basic/
cleansed levels, a renamed-twin case where an exact-name match resolves to
the wrong target and only scanning every source rather than skipping the
already-matched ones reaches the true match, a near-name family, a
duplicate target name, a genuine `never`-level pair, an unlabelled source and
a source with no candidate at all -- see that file's own `cases` for each
one's story). This module builds a matched and canonical layer pair from those
rows once per test run and scores it through `execute_blocking_run()` once
for every `(representation, similarity_backend)` cell this area supports
without an optional extra, plus the `sbert` cells when the optional
`sentence-transformers` extra is importable, asserting each case's expected
cluster, the exact-join cascade's per-form counts, and the name-equality
level each labelled pair is attributed to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from company_tokenize.training import train_tokenizer_with_trainer

from blocking.contracts import BlockingRunConfig, BlockingStrategyConfig
from blocking.loader import load_dataset_descriptor
from blocking.workflow import execute_blocking_run
from tests.promoted_tokenizers import promoted_tokenizer_files
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model_pairing.json"
FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
SOURCE_SYSTEM: str = FIXTURE["source_system"]
TARGET_SYSTEM: str = FIXTURE["target_system"]
COUNTRY: str = FIXTURE["country"]
TARGETS: list[dict[str, Any]] = FIXTURE["targets"]
SOURCES: list[dict[str, Any]] = FIXTURE["sources"]

# Every representation and similarity backend this area's own CLI surface
# selects (`blocking._cli_helper.SIMILARITY_BACKENDS`), minus `hdbscan`:
# `test_workflow.py` no longer carries even a mocked hdbscan test, since the
# backend "isn't wired for any representation this area runs today" (this
# package's README, Known Issues) -- sklearn's HDBSCAN has no sparse-input
# support and this area never builds a dense target for a sparse
# representation, so every cell this suite could add would either refuse or
# need the same mocking that test was retired for.
_SPARSE_REPRESENTATIONS: tuple[str, ...] = ("tfidf", "wordpiece", "sentencepiece")
_SPARSE_BACKENDS: tuple[str, ...] = (
    "sklearn",
    "sparse_dot_topn",
    "svd_rerank",
    "kmeans",
)
# `dense_brute`/`hnsw` refuse a sparse target matrix outright, and `sbert` is
# the only representation this area builds a dense target from; sbert's own
# strategy refuses `sparse_dot_topn`/`svd_rerank` (both assume a sparse
# target), see `company_vectorize.dense_encoder_strategy.UNSUPPORTED_DENSE_BACKENDS`.
_SBERT_BACKENDS: tuple[str, ...] = ("sklearn", "kmeans", "dense_brute", "hnsw")


# `sentence_transformers` is never imported at module level here, only
# through `pytest.importorskip` inside the test itself. Importing it
# in an otherwise-empty pytest process crashes the whole process with a
# Windows `STATUS_HEAP_CORRUPTION` fault (`0xc0000374`) -- reproducible with
# nothing more than `pytest.importorskip("sentence_transformers")`, and not a
# pytest bug: the same bare `python -c "import sentence_transformers"`, with
# nothing imported first, crashes identically outside pytest entirely.
# `sentence_transformers` pulls in `pandas`, which pulls in `pyarrow`;
# loading `pyarrow`'s native extension is what faults, and whether that fault
# is the unrecoverable heap-corruption kind or a benign, recoverable one
# depends on the process's native-heap layout at the point it loads --
# shifted just by which native extensions already sit in the process
# (`sklearn`, `pandas` and `psutil` all move it; bare `numpy`/`scipy` alone do
# not). This module's own `from blocking.workflow import execute_blocking_run`
# above already pulls in `sklearn`/`pandas`/`psutil` for the sparse
# representations' backends, so by the time any test here calls
# `importorskip`, the crash cannot occur -- confirmed both as a plain
# `python -c` repro and under this exact pytest/tach/cov plugin chain. The
# cause is a real, upstream native-library issue outside this repository's
# control; this module is simply never the isolated-import case that
# triggers it.
CELLS: list[Any] = [
    (representation, backend)
    for representation in _SPARSE_REPRESENTATIONS
    for backend in _SPARSE_BACKENDS
] + [("sbert", backend) for backend in _SBERT_BACKENDS]


def _write_partition(
    roots: WorkspaceRoots, *, system: str, layer: str, rows: list[dict[str, object]]
) -> Path:
    layer_dir = system_layer_dir(roots, system, layer=layer)
    if layer == CANONICAL_LAYER_NAME:
        # A target is read from a dated snapshot.
        layer_dir = layer_dir / "2026-01-01"
    partition_dir = layer_partition_dir(layer_dir, value=COUNTRY)
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(partition_dir / "part-00001.parquet")
    return layer_dir


def _write_match_metadata(
    layer_dir: Path, *, source_system: str, target_systems: list[str]
) -> None:
    metadata = {"source_system": source_system, "target_systems": target_systems}
    (layer_dir / "_match_metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )


def _train_tiny_tokenizers(roots: WorkspaceRoots) -> None:
    """Train a real, tiny WordPiece and SentencePiece tokenizer for each
    system directly from this fixture's own vocabulary, so the `wordpiece`/
    `sentencepiece` representations tokenize meaningfully instead of reading
    whatever real, much larger tokenizer this machine happens to have
    trained under `artifacts/tokenizers/` (a tracked directory this test
    never reads from or writes to -- see `worktree-setup`'s write policy).
    Both systems train on the identical combined corpus so a shared token
    like "consulting" or "ltd" is not fragmented differently on each side.
    """
    names = [row["name"].lower() for row in (*TARGETS, *SOURCES)]
    corpus_path = roots.checkout / "model_pairing_corpus.parquet"
    pl.DataFrame({"name": names}).write_parquet(corpus_path)

    for system in (SOURCE_SYSTEM, TARGET_SYSTEM):
        for trainer in ("wordpiece", "sentencepiece"):
            train_tokenizer_with_trainer(
                trainer=trainer,
                corpus_path=corpus_path,
                tokenizer_path=promoted_tokenizer_files(
                    roots, system=system, trainer=trainer
                ).model,
                vocab_size=200,
                show_progress=False,
            )


def _build_layer_fixture(roots: WorkspaceRoots) -> None:
    matched_rows = [
        {
            "system_uri": f"{SOURCE_SYSTEM}:{row['id']}",
            "name": row["name"],
            "jurisdiction_code": COUNTRY,
            "match_uri": (
                f"{TARGET_SYSTEM}:{row['match_target']}"
                if row["match_target"]
                else None
            ),
        }
        for row in SOURCES
    ]
    matched_dir = _write_partition(
        roots, system=SOURCE_SYSTEM, layer=MATCHED_LAYER_NAME, rows=matched_rows
    )
    _write_match_metadata(
        matched_dir, source_system=SOURCE_SYSTEM, target_systems=[TARGET_SYSTEM]
    )

    target_rows = [
        {
            "system_uri": f"{TARGET_SYSTEM}:{row['id']}",
            "name": row["name"],
            "jurisdiction_code": COUNTRY,
        }
        for row in TARGETS
    ]
    _write_partition(
        roots, system=TARGET_SYSTEM, layer=CANONICAL_LAYER_NAME, rows=target_rows
    )

    _train_tiny_tokenizers(roots)


def _build_config(
    roots: WorkspaceRoots, *, representation: str, similarity_backend: str
) -> BlockingRunConfig:
    source = load_dataset_descriptor(roots=roots, system=SOURCE_SYSTEM)
    target = load_dataset_descriptor(
        roots=roots, system=TARGET_SYSTEM, require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation=representation,
        top_k=2,
        # Higher than most fixtures in this area (0.2): this fixture's
        # vocabulary deliberately reuses generic legal-suffix and filler
        # words ("Ltd", "Group", "Consulting") across unrelated cases, and
        # at this population's tiny size a low threshold lets those shared
        # substrings alone clear it, chaining cases together that the
        # fixture means to keep separate. Verified against tfidf/sklearn's
        # own raw_matched_edges: every case's real signal scores >= 0.7,
        # every unintended cross-case echo scores <= 0.4.
        min_similarity=0.5,
        max_candidates_per_source=None,
        similarity_backend=similarity_backend,
        tokenizer="sentencepiece" if representation == "sentencepiece" else "wordpiece",
        # Linking is on for every cell, so the fixture's near_family case
        # exercises target-to-target linking rather than only the
        # source-to-target scan. 0.45 is picked directly against this
        # fixture's own raw target-neighbor similarities (not assumed): every
        # in-family pairing (near_family's t6/t7, near_family_over_cap's
        # t12-t15) scores >= 0.497 under every representation here, and the
        # one unintended cross-family echo (t6/t7 against t12/t15) never
        # exceeds 0.38.
        target_neighbor_min_similarity=0.45,
        target_neighbor_max_per_target=2,
    )
    return BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
    )


def _cluster_map(clusters: pl.DataFrame) -> dict[str, str]:
    """`node_id` -> `cluster_id`, over every clustered node (source or target)."""
    return dict(
        zip(
            clusters.get_column("node_id").to_list(),
            clusters.get_column("cluster_id").to_list(),
            strict=True,
        )
    )


@pytest.mark.integration
@pytest.mark.parametrize(("representation", "similarity_backend"), CELLS)
def test_model_pairing_known_answer(
    workspace_roots: WorkspaceRoots, representation: str, similarity_backend: str
) -> None:
    if representation == "sbert":
        pytest.importorskip("sentence_transformers")
    _build_layer_fixture(workspace_roots)
    config = _build_config(
        workspace_roots,
        representation=representation,
        similarity_backend=similarity_backend,
    )

    result = execute_blocking_run(config)

    node_cluster = _cluster_map(result.clusters)

    for source in SOURCES:
        source_node = f"{SOURCE_SYSTEM}:{source['id']}"
        case = source["case"]
        override = source["divergence_notes"].get(similarity_backend)
        applies = override is not None and representation in override.get(
            "representations", [representation]
        )
        expected_ids = (
            override["expected_cluster_targets"]
            if applies
            else source["expected_cluster_targets"]
        )
        expected_targets = [f"{TARGET_SYSTEM}:{t}" for t in expected_ids]

        if case == "no_candidate":
            assert source_node not in node_cluster, (
                f"case={case} source={source['id']} representation={representation} "
                f"backend={similarity_backend}: expected no candidate at all, but it "
                f"landed in cluster {node_cluster.get(source_node)!r}"
            )
            continue

        assert source_node in node_cluster, (
            f"case={case} source={source['id']} representation={representation} "
            f"backend={similarity_backend}: expected a cluster, found none"
        )
        source_cluster = node_cluster[source_node]

        if case == "near_family" and similarity_backend == "kmeans":
            # kmeans routes a probed target to its single nearest partition's
            # centroid the same way it routes a scanned source
            # (company_vectorize.partition_similarity) -- there is no
            # sub-linear *nearest-neighbour* backend here, only a
            # nearest-*partition* one, so the target-neighbour probe
            # inherits the same "may not reach every true near neighbour"
            # limit `s4`'s own kmeans/tfidf override already documents.
            # Verified 2026-09-14: every kmeans cell links the source to
            # exactly one family member (which one varies by
            # representation), never both -- a real backend limit, not a
            # fixture bug, so the assertion stays at "connectivity to *a*
            # family member" for this backend only, as it did before linking.
            assert any(
                node_cluster.get(t) == source_cluster for t in expected_targets
            ), (
                f"case={case} source={source['id']} representation={representation} "
                f"backend={similarity_backend}: expected cluster {source_cluster!r} to "
                f"contain at least one of {expected_targets}, actual members: "
                f"{[n for n, c in node_cluster.items() if c == source_cluster]}"
            )
            continue

        # Every other case, including near_family on every backend
        # but kmeans, expects the whole named cluster -- the source's own
        # scan reaches at least one family member (which one is allowed to
        # differ by representation, per the fixture's near_family note), and
        # the target-neighbour edges this run's own target index was probed
        # for then link every family member to the others, so the whole
        # family is one cluster regardless of which member the scan itself
        # reached.
        for target_node in expected_targets:
            assert node_cluster.get(target_node) == source_cluster, (
                f"case={case} source={source['id']} representation={representation} "
                f"backend={similarity_backend}: expected {target_node} in cluster "
                f"{source_cluster!r}, actual members: "
                f"{[n for n, c in node_cluster.items() if c == source_cluster]}"
            )

    # -- Per-form counts (the exact-join cascade) --------------------
    summary = result.exact_match_summary.row(0, named=True)
    # s1 (raw), s4 and s6/duplicate-target (raw, via the decoy/duplicate)
    # resolve at the raw level; s2 at basic; s3 at cleansed.
    assert summary["resolved_by_raw"] >= 3
    assert summary["resolved_by_basic"] >= 1
    assert summary["resolved_by_cleansed"] >= 1
    # s6's raw form maps to both t8 and t9 (`multi_target_source_count`).
    assert summary["multi_target_source_count"] >= 1

    # -- Level attribution (the per-pair detail) ----------------------
    assert result.pair_truth_eval_detail is not None
    detail = result.pair_truth_eval_detail.filter(pl.col("is_truth_pair"))
    levels_by_source = dict(
        zip(
            detail.get_column("source_id").to_list(),
            detail.get_column("name_equality").to_list(),
            strict=True,
        )
    )
    assert levels_by_source[f"{SOURCE_SYSTEM}:s1"] == "raw"
    assert levels_by_source[f"{SOURCE_SYSTEM}:s2"] == "basic"
    assert levels_by_source[f"{SOURCE_SYSTEM}:s3"] == "cleansed"
    assert levels_by_source[f"{SOURCE_SYSTEM}:s7"] == "never"

    # -- Target-neighbour linking ------------------------------------------
    # near_family's t6/t7 land in one cluster above, whichever member the
    # scan itself reached -- this only reads back the probe that made that
    # possible. near_family_over_cap's t12-t15 carry no source at all: each
    # has three other family members to link to against this test's
    # target_neighbor_max_per_target=2, so a sub-linear *nearest-neighbour*
    # backend keeps every one of the four at the cap, which is what a run
    # above the cap being *reported* (rather than silently truncated with
    # nothing to show for it) means. kmeans is excluded: its own probe
    # routes each queried target to its single nearest *partition*, not its
    # true nearest neighbours, so a partition well short of the cap never
    # truncates anything -- the same routing limit `near_family`'s own
    # kmeans branch above already documents, reapplied here since it is the
    # same backend property, not a second coincidence.
    assert result.target_neighbor_summary.height == 1
    neighbor_summary = result.target_neighbor_summary.row(0, named=True)
    if similarity_backend != "kmeans":
        assert neighbor_summary["targets_at_cap"] >= 4
        over_cap_targets = {f"{TARGET_SYSTEM}:t{n}" for n in (12, 13, 14, 15)}
        edges = result.target_neighbor_edges
        at_cap_ids = set(
            edges.filter(pl.col("target_id_a").is_in(over_cap_targets))
            .get_column("target_id_a")
            .unique()
            .to_list()
        )
        assert over_cap_targets <= at_cap_ids, (
            f"representation={representation} backend={similarity_backend}: expected "
            f"every near_family_over_cap target to probe at least one neighbour, "
            f"got edges from {at_cap_ids} only"
        )
