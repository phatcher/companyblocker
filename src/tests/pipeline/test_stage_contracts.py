"""One corpus driven through every stage, each reading its predecessor's real output.

The per-stage suites each write the input their own stage expects, which
leaves the handoffs between stages asserted nowhere: a stage can change the
shape of what it emits and every one of those tests still passes. That gap
has already reached real data. A canonical snapshot that was flat, for
systems whose canonical layer partitions by `jurisdiction_code`, would have
replaced a partitioned cleansed view with a flat one; nothing in the suite
caught it, and the dry run did not either, because staleness there compares
run dates rather than data shape. It was caught at runtime by a guard
hand-written into one stage.

So these tests assert the contract, not the values. Each boundary is
checked by handing the upstream stage's real output to the downstream
stage's *own* resolver -- `workspace.layer_layout` for layer shape,
`workspace.cleanse_inputs` for what Cleanse selects, `blocking.loader` for
what a blocking run resolves -- rather than by re-deriving inside the test
what the schema and paths ought to be. A test that re-derives them is a
second copy of the rule, and drifts from the stages independently.

**Which stages are here, and why not `tokenize`.** Shard, Canonical,
Cleanse, Match and the blocking run form one chain: each writes a layer the
next one reads, so each pair is a contract a change can break silently.
Tokenize is not part of that chain. No stage in it reads `tokenized/` --
a blocking run's `cluster_tokens` feature column is derived in memory by
`company_vectorize.tfidf_cluster`, not loaded from the tokenized layer -- so
tokenize has no downstream consumer here whose input contract could break.
Its only readers are in `src/analysis`, a different boundary set, and it
additionally needs a trained tokenizer, which is an artifact of the training
flow rather than anything a corpus of company records carries. Covering it
would mean adding a training dependency to a data-contract test in exchange
for a boundary this chain does not contain.

Match, by contrast, needs only a second system cleansed into the same
jurisdiction, which a two-system corpus carries anyway because a blocking
run needs two datasets. It is also the stage that writes the `matched/`
layer `blocking.loader` prefers, so leaving it out would mean the blocking
boundary was only ever tested against the fallback layer.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from acquisition import canonical
from acquisition.cleansed_contracts import CLEANSED_MERGE_REQUIRED_COLUMNS
from acquisition.match_ops import materialize_match_uri_artifact
from acquisition.registry import get_system_plan
from blocking.contracts import (
    BlockingRunConfig,
    BlockingStrategyConfig,
    validate_blocking_artifact_schema,
)
from blocking.loader import (
    load_country_frame,
    load_dataset_descriptor,
    load_name_variant_frame,
)
from blocking.workflow import execute_blocking_run
from scripts import process_companies
from tests.pipeline.corpus import PipelineCorpus
from validation.contracts import validate_artifact_schema
from workspace.cleanse_inputs import resolve_input_dir
from workspace.layer_layout import (
    partition_values,
    primary_family_dir,
    resolve_name_files,
    resolve_primary_files,
)

pytestmark = pytest.mark.integration


def _run_stages(
    corpus: PipelineCorpus, *, systems: list[str], stages: list[str]
) -> int:
    """Drive real pipeline stages over the corpus, as the CLI drives them."""
    return process_companies.run_pipeline(
        systems=systems,
        processes=stages,
        root=corpus.root,
        run_date=corpus.run_date,
        force=True,
    )


def _relative_primary_paths(layer_dir: Path, *, system: str) -> set[str]:
    """Each primary-family file's path relative to its layer root.

    The mirroring contract is expressed in exactly these terms -- a mapping
    stage reproduces its input's own relative path under its own root -- so
    comparing two layers' sets of them compares the contract itself rather
    than a reconstruction of it.
    """
    return {
        path.relative_to(layer_dir).as_posix()
        for path in resolve_primary_files(layer_dir, system_code=system)
    }


def _blocking_config(corpus: PipelineCorpus) -> BlockingRunConfig:
    source = load_dataset_descriptor(roots=corpus.roots, system=corpus.source_system)
    target = load_dataset_descriptor(
        roots=corpus.roots, system=corpus.target_system, require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.1,
        max_candidates_per_source=None,
        similarity_backend="sklearn",
        backend_options=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
        text_view=None,
        tokenizer="wordpiece",
        max_candidates_per_target=None,
        candidate_similarity_ratio=None,
    )
    return BlockingRunConfig(
        roots=corpus.roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
    )


def test_each_stage_consumes_the_previous_stage_s_real_output(
    pipeline_corpus: PipelineCorpus,
) -> None:
    """Shard, Canonical, Cleanse, Match and a blocking run over one corpus.

    Fails if any stage stops being able to consume what its predecessor
    wrote: the stages report a non-zero exit on a rejected input, and the
    shape assertions between them catch the worse case, where a stage
    accepts its input but emits a differently-shaped layer than the next one
    expects.
    """
    corpus = pipeline_corpus
    source_system = corpus.source_system
    target_system = corpus.target_system

    # --- Acquire -> Shard ---------------------------------------------
    assert (
        _run_stages(corpus, systems=[source_system, target_system], stages=["shard"])
        == 0
    )
    for system in (source_system, target_system):
        shard_dir = corpus.source_root(system)
        assert resolve_primary_files(shard_dir, system_code=system), (
            f"Shard wrote no primary files for '{system}' that the layout "
            "resolver recognizes, so Canonical would find none either."
        )

    # --- Shard -> Canonical -------------------------------------------
    assert (
        _run_stages(
            corpus, systems=[source_system, target_system], stages=["canonical"]
        )
        == 0
    )
    canonical_dir = corpus.canonical_dir(source_system)
    assert partition_values(canonical_dir) == list(corpus.source_jurisdictions), (
        "Canonical must partition a multi-jurisdiction system by jurisdiction, "
        "which is the shape every layer below it mirrors."
    )
    assert resolve_name_files(canonical_dir, system_code=source_system), (
        "Canonical dropped the name-variant family the name-variant "
        "recovery run scores against a system's own primary records."
    )

    # --- Canonical -> Cleanse -----------------------------------------
    # Cleanse's own input resolver, not a path rebuilt here, is what has to
    # find the snapshot Canonical just wrote.
    assert (
        resolve_input_dir(roots=corpus.roots, run_date=None, system=source_system)
        == canonical_dir
    )
    assert (
        _run_stages(corpus, systems=[source_system, target_system], stages=["cleanse"])
        == 0
    )

    cleansed_dir = corpus.layer_dir(source_system, "cleansed")
    assert partition_values(cleansed_dir) == partition_values(canonical_dir)
    assert _relative_primary_paths(
        primary_family_dir(cleansed_dir), system=source_system
    ) == _relative_primary_paths(
        primary_family_dir(canonical_dir), system=source_system
    ), "Cleanse mirrors Canonical's layout; it does not choose its own."

    canonical_frame = pl.read_parquet(
        resolve_primary_files(canonical_dir, system_code=source_system)
    )
    cleansed_frame = pl.read_parquet(
        resolve_primary_files(cleansed_dir, system_code=source_system)
    )
    assert cleansed_frame.height == canonical_frame.height, (
        "Cleanse maps canonical rows one-to-one; a row count change means it "
        "merged or dropped, which downstream row-level joins do not expect."
    )
    assert set(canonical_frame.columns) <= set(cleansed_frame.columns), (
        "Cleanse adds derived columns and must not drop canonical ones."
    )
    assert set(CLEANSED_MERGE_REQUIRED_COLUMNS) <= set(cleansed_frame.columns)

    # --- Canonical -> Match -------------------------------------------
    # Match joins on canonical-schema fields, so it reads the snapshot
    # Canonical wrote and not the cleansed layer derived from it.
    # The source and the target are both named: a match is this source
    # labelled against that target, and writes the source's matched layer.
    #
    # Called at the stage's own entrypoint rather than through the pipeline
    # runner: Canonical's identity write removed the name-variant enrichment step Match used to
    # pair with (a later stage mutating Canonical's own output), so there is
    # no longer a second step to call through the runner for.
    summary = materialize_match_uri_artifact(
        roots=corpus.roots, source_system=source_system, target_system=target_system
    )
    assert (summary.source_system, summary.target_system) == (
        source_system,
        target_system,
    )

    matched_dir = corpus.layer_dir(source_system, "matched")
    assert partition_values(matched_dir) == [corpus.shared_jurisdiction], (
        "Match writes only the jurisdiction both systems share, in the same "
        "partitioned shape as the canonical snapshot it derives from."
    )
    matched_frame = pl.read_parquet(
        resolve_primary_files(matched_dir, system_code=source_system)
    )
    assert set(matched_frame.columns) == set(canonical_frame.columns), (
        "Match fills in canonical's own `match_uri` and carries no column "
        "Cleanse derives, which would age there."
    )
    assert matched_frame.get_column("match_uri").drop_nulls().len() >= 1, (
        "Match resolved no ground-truth pair, so the blocking boundary below "
        "would be scored against nothing."
    )

    # --- Match/Canonical -> blocking run ------------------------------
    config = _blocking_config(corpus)
    assert config.source.layer == "matched"
    assert config.source.has_ground_truth
    assert config.target.layer == "canonical"
    assert corpus.shared_jurisdiction in config.source.available_countries
    assert corpus.shared_jurisdiction in config.target.available_countries

    for descriptor in (config.source, config.target):
        frame = load_country_frame(descriptor, country=corpus.shared_jurisdiction)
        assert frame.height >= 1
        # The raw name is all a run needs: it derives every name form itself.
        assert {"system_uri", "name"} <= set(frame.columns)

    assert load_name_variant_frame(config.target) is not None, (
        "The name-variant recovery run (a separate labelled run from "
        "this canonical blocking run) needs Canonical's name-variant family "
        "to be loadable through the blocking loader."
    )

    result = execute_blocking_run(config)

    validate_blocking_artifact_schema(
        result.matched_edges, artifact_name="matched_edges"
    )
    assert result.matched_edges.height >= 1
    assert result.pair_truth_eval is not None
    # Truth scoring is the validation package's artifact, not blocking's, so
    # its own registry is what accepts it.
    validate_artifact_schema(result.pair_truth_eval, artifact_name="pair_truth_eval")
    assert result.pair_truth_eval.get_column("labelled_sources").sum() >= 1, (
        "The blocking run saw no labelled source, which means the match_uri "
        "Match wrote did not survive into the layer the run scored."
    )


def test_cleanse_refuses_a_flat_canonical_layer_for_a_partitioning_system(
    pipeline_corpus: PipelineCorpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape mismatch that reached real data, reproduced through the stages.

    A canonical snapshot written before partitioning moved into the Canonical
    stage is flat, and mapping it one-to-one would replace the partitioned
    cleansed view with a flat one. Here the real Canonical stage writes that
    snapshot, because the resource it is given declares no partition column
    -- the same configuration a pre-partitioning run had -- and Cleanse then
    reads it with the system's declared partition column still in force, as
    it would in production.

    Producing the flat shape rather than asserting the guard's message is
    what makes this a contract test: it fails if either stage stops agreeing
    with the other about what a partitioned layer is, not only if the guard's
    wording changes.
    """
    corpus = pipeline_corpus
    system = corpus.source_system

    assert _run_stages(corpus, systems=[system], stages=["shard"]) == 0

    plan = get_system_plan(system)
    unpartitioned_plan = replace(
        plan,
        resources=(
            replace(plan.resources[0], output_defaults={"partition_by": None}),
            *plan.resources[1:],
        ),
    )
    # Scoped to the Canonical run alone: Cleanse below has to resolve the
    # system's *declared* partition column, which is what makes the two
    # stages disagree.
    with monkeypatch.context() as unpartitioned:
        unpartitioned.setattr(
            canonical, "get_system_plan", lambda _: unpartitioned_plan
        )
        assert _run_stages(corpus, systems=[system], stages=["canonical"]) == 0

    canonical_dir = corpus.canonical_dir(system)
    assert partition_values(canonical_dir) == [], (
        "The canonical snapshot this case exists for is a flat one; if "
        "Canonical partitioned it anyway, Cleanse is not being handed the "
        "shape that reached real data."
    )
    assert resolve_primary_files(canonical_dir, system_code=system), (
        "The flat snapshot still has to be real, readable canonical output "
        "-- otherwise Cleanse would refuse it for being empty instead."
    )

    assert _run_stages(corpus, systems=[system], stages=["cleanse"]) == 1
    cleansed_dir = corpus.layer_dir(system, "cleansed")
    assert not resolve_primary_files(cleansed_dir, system_code=system), (
        "Cleanse refused the input but still wrote a cleansed layer, which "
        "is the flat-for-partitioned replacement the refusal exists to stop."
    )
