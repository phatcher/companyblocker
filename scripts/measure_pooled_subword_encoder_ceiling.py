"""Measure the pooled-subword contrastive encoder's ceiling on real pairs.

Trains `company_classify.PooledSubwordContrastiveEncoder` (via `cross_train_pairs`) over the
same real observed name-variant pairs, entity-level split and negative sampling
`measure_pair_classifier_ceiling.py` uses for the dedicated pair classifier's own measured
ceiling, so the two are directly comparable, `MetricBundle` for `MetricBundle`, against a shared
baseline rather than a separately-tuned one.

The encoder pools over an already-trained WordPiece tokenizer. No such artifact exists under
`data/` yet, so this script trains one itself, from the same real names this run already loads,
writing only under `--corpus-out` (`tmp/` by default) -- this never writes under `data/`, and
never re-runs acquire/shard, matching `measure_pair_classifier_ceiling.py`'s own scope.

This also measures a pretrained-S-BERT baseline (`SbertPretrainedPairScorer`): the same
checkpoint `company_vectorize.sbert_strategy`'s clustering strategy defaults to, scored over the
identical pairs/split via `cross_train_pairs()` rather than through that strategy's
clustering/candidate-generation path, which answers a different question (top-k candidates, not
a pair's match score). The `sentence-transformers` optional dependency is not installed in this
repo's shared `.venv` by default (it resolves in `uv.lock` only transitively, through
`company_vectorize`'s own `sbert` extra); pass `--skip-sbert` to omit that half of the report
when it is not synced, rather than have the whole run fail after the slower tokenizer/encoder
training already ran.

Usage:
    .venv/Scripts/python.exe scripts/measure_pooled_subword_encoder_ceiling.py \
        --json-out tmp/pooled_subword_encoder_ceiling/wikidata_gb_encoder_ceiling.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    RoleSetting,
    add_declared_arguments,
    add_dry_run_arg,
    add_role_settings,
    add_workspace_roots_args,
    declared_settings,
    report_output_plan,
    report_resolved_settings,
    resolve_declared_settings,
    resolve_role_settings,
    resolve_workspace_roots_from_args,
    resolved_setting_values,
    run_reporting_argument_errors,
)
from company_classify import (
    EntitySplitter,
    NameVariant,
    NegativeSamplingConfig,
    PooledSubwordContrastiveEncoder,
    RealAliasPairProducer,
    TfidfPairMlpClassifier,
    cross_train_pairs,
    split_pairs,
    validate_pairs,
)
from company_classify._cli_helper import SETTINGS as CLASSIFY_SETTINGS
from company_tokenize import load_tokenizer_vocabulary, train_wordpiece
from company_vectorize import (
    DEFAULT_SBERT_MODEL_NAME,
    ensure_sentence_embedding_checkpoint,
    resolve_sbert_model_name,
)
from company_vectorize._cli_helper import SETTINGS as VECTORIZE_SETTINGS
from measure_pair_classifier_ceiling import (
    SETTING_SURFACE as _CLASSIFY_SETTING_SURFACE,
)
from measure_pair_classifier_ceiling import (
    load_real_alias_variants,
    resolve_relative_to_checkout,
    resolve_splitter_and_negatives,
)

from workspace.roots import WorkspaceRoots

_SYSTEM = "wikidata"
_CANONICAL_DATE = "2026-07-16"
_JURISDICTION = "GB"

DECLARATIONS = declared_settings(CLASSIFY_SETTINGS, VECTORIZE_SETTINGS)
"""This script's own settings plus the sibling classifier script's, so both
resolve the same split/negative-sampling values the same way."""

SETTING_SURFACE: dict[str, dict[str, object]] = {
    **_CLASSIFY_SETTING_SURFACE,
    "sbert_model_name": {
        "flag": "--sbert-model",
        "default": DEFAULT_SBERT_MODEL_NAME,
        "note": (
            "Registry slug, hub checkpoint id, or local checkpoint path, resolved the "
            "same way SbertClusteringStrategy resolves its own model_name. "
            "Ignored if --skip-sbert is set."
        ),
    },
}


def _load_sentence_transformer_checkpoint(checkpoint: str) -> Any:
    """Load a real `sentence_transformers.SentenceTransformer` checkpoint.

    The default `SbertPretrainedPairScorer.model_factory`. Imports `sentence_transformers`
    lazily (only when a real load is actually requested) and runs the same pooling-checkpoint
    gate `SbertClusteringStrategy` runs (`sbert_pooling_gate.py`), so a checkpoint with no
    declared sentence-embedding pooling config fails the same way it would there.
    """
    from sentence_transformers import SentenceTransformer

    ensure_sentence_embedding_checkpoint(checkpoint)
    return SentenceTransformer(checkpoint)


@dataclass
class SbertPretrainedPairScorer:
    """Pretrained-S-BERT pair scorer: `SbertClusteringStrategy`'s default checkpoint, scored like a pair classifier.

    Implements `pair_models.PairClassifierModel` so it slots into `cross_train_pairs()`
    unmodified, `MetricBundle` for `MetricBundle`, against
    `PooledSubwordContrastiveEncoder`/`TfidfPairMlpClassifier`. `fit()` is a no-op: the checkpoint
    is pretrained here, not fine-tuned on these pairs -- fine-tuning it is a separate, larger
    change with its own training loop.
    `predict_scores()` maps a pair's cosine similarity into `[0, 1]` the same way
    `PooledSubwordContrastiveEncoder.predict_scores()` does, so the two encoders' scores land on
    the same scale under `evaluate.compute_binary_metrics`'s default 0.5 threshold.

    Loads `sentence_transformers.SentenceTransformer` lazily, on first `predict_scores()` call
    rather than at import time, so importing this module (and the rest of this script's report)
    does not require the optional dependency to be installed.

    Attributes:
        model_name: A `company_vectorize` registry slug, hub checkpoint id, or local checkpoint
            path, resolved via `resolve_sbert_model_name()` exactly as
            `SbertClusteringStrategy` resolves its own `model_name` -- so "the pretrained S-BERT
            baseline" means the same checkpoint that strategy would load, not a separately
            chosen one.
        model_factory: Builds the loaded model given the resolved checkpoint. Defaults to a real
            `sentence_transformers.SentenceTransformer`, imported lazily inside this factory
            (not at module import time) so importing this module never requires the optional
            dependency. Tests inject a fake factory here instead of monkeypatching
            `sentence_transformers` -- the same seam `SbertClusteringStrategy.encoder_factory`
            uses for the same reason.
    """

    model_name: str = DEFAULT_SBERT_MODEL_NAME
    model_factory: Callable[[str], Any] = field(default=None, repr=False)  # type: ignore[assignment]
    _model: Any = field(init=False, repr=False, default=None)
    _vector_cache: dict[str, np.ndarray] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.model_factory is None:
            self.model_factory = _load_sentence_transformer_checkpoint

    def fit(self, pairs: Sequence[Any]) -> None:
        _ = pairs

    def _model_instance(self) -> Any:
        if self._model is None:
            checkpoint = resolve_sbert_model_name(self.model_name)
            self._model = self.model_factory(checkpoint)
        return self._model

    def _ensure_embedded(self, names: list[str]) -> None:
        unseen = sorted({name for name in names if name not in self._vector_cache})
        if not unseen:
            return
        vectors = self._model_instance().encode(unseen, normalize_embeddings=True)
        for name, vector in zip(unseen, vectors, strict=True):
            self._vector_cache[name] = np.asarray(vector, dtype=np.float64)

    def predict_scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self._ensure_embedded([name for pair in pairs for name in pair])
        scores: list[float] = []
        for left_name, right_name in pairs:
            cosine = float(
                np.dot(self._vector_cache[left_name], self._vector_cache[right_name])
            )
            scores.append((cosine + 1.0) / 2.0)
        return scores


def train_tokenizer_over_names(
    names: list[str], *, corpus_out: Path, tokenizer_out: Path
) -> Path:
    """Train a WordPiece tokenizer over `names`, writing the corpus and artifact under `corpus_out`/`tokenizer_out`.

    A prepared parquet corpus is `train_wordpiece()`'s whole contract (one `name` column), so
    this writes exactly that from the same real names the run already loaded, rather than
    reaching into the tokenize stage's own pipeline for a corpus-preparation step this
    measurement does not otherwise need.
    """
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name": names}).write_parquet(corpus_out)
    tokenizer_out.parent.mkdir(parents=True, exist_ok=True)
    train_wordpiece(
        corpus_path=corpus_out, tokenizer_path=tokenizer_out, show_progress=False
    )
    return tokenizer_out


def build_ceiling_report(
    *,
    system: str,
    jurisdiction_code: str,
    canonical_date: str,
    splitter: EntitySplitter,
    negatives: NegativeSamplingConfig,
    variants: list[NameVariant],
    tokenizer_path: Path,
    sbert_model_name: str | None = DEFAULT_SBERT_MODEL_NAME,
    sbert_model_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Run the pooled-subword encoder, the dedicated pair classifier, and (unless omitted) the pretrained S-BERT baseline over `variants`, same pairs and split.

    `sbert_model_name=None` skips the S-BERT baseline entirely (`--skip-sbert`), for a run where
    the optional `sentence-transformers` dependency is not synced -- the report then omits the
    `"sbert_baseline"` key and the `ceiling`'s `encoder_vs_sbert_*` entries rather than fail after
    the slower tokenizer/encoder training already ran.

    `sbert_model_factory` overrides `SbertPretrainedPairScorer.model_factory` (a fake, in a
    test, so the S-BERT baseline can be exercised without the optional `sentence-transformers`
    dependency installed); `None` means the real checkpoint loader.

    Raises `PairContractError` (via `validate_pairs()`) if the generated pairs break the
    split-leakage, structural, or ambiguity contract `pairs.py` defines.
    """
    producer = RealAliasPairProducer(
        variants=variants, splitter=splitter, negatives=negatives
    )
    pairs = producer.generate()
    validate_pairs(pairs, splitter=splitter)

    split = split_pairs(pairs)

    vocabulary = load_tokenizer_vocabulary(tokenizer_path, trainer="wordpiece")
    encoder = PooledSubwordContrastiveEncoder(
        vocab=dict(vocabulary.vocab),
        encode=vocabulary.encode,
        unk_token=vocabulary.unk_token,
    )

    encoder_start = time.perf_counter()
    encoder_results = cross_train_pairs(split, {"pooled_subword": encoder})
    encoder_elapsed = time.perf_counter() - encoder_start

    classifier_start = time.perf_counter()
    classifier_results = cross_train_pairs(
        split, {"pair_mlp": TfidfPairMlpClassifier()}
    )
    classifier_elapsed = time.perf_counter() - classifier_start

    encoder_metrics = encoder_results["pooled_subword"].test_metrics
    classifier_metrics = classifier_results["pair_mlp"].test_metrics

    fit_seconds = {
        "pooled_subword_encoder": encoder_elapsed,
        "pair_classifier": classifier_elapsed,
    }

    report: dict[str, Any] = {
        "run": {
            "system": system,
            "jurisdiction_code": jurisdiction_code,
            "canonical_date": canonical_date,
            "producer": "RealAliasPairProducer",
            "tokenizer_vocab_size": len(vocabulary.vocab),
            "split_seed": splitter.seed,
            "split_ratios": {
                "train": splitter.ratios.train,
                "validation": splitter.ratios.validation,
                "test": splitter.ratios.test,
            },
            "negative_strategy": negatives.strategy.value,
            "negatives_per_positive": negatives.negatives_per_positive,
            "negative_seed": negatives.seed,
            "variant_rows": len(variants),
            "pair_counts": {
                "total": len(pairs),
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
            "fit_seconds": fit_seconds,
        },
        "pooled_subword_encoder": {
            "name": "pooled_subword",
            "validation_metrics": asdict(
                encoder_results["pooled_subword"].validation_metrics
            ),
            "test_metrics": asdict(encoder_metrics),
        },
        "pair_classifier": {
            "name": "pair_mlp",
            "validation_metrics": asdict(
                classifier_results["pair_mlp"].validation_metrics
            ),
            "test_metrics": asdict(classifier_metrics),
        },
        "ceiling": {
            "encoder_f1": encoder_metrics.f1,
            "classifier_f1": classifier_metrics.f1,
            "f1_delta": encoder_metrics.f1 - classifier_metrics.f1,
            "encoder_pr_auc": encoder_metrics.pr_auc,
            "classifier_pr_auc": classifier_metrics.pr_auc,
            "pr_auc_delta": encoder_metrics.pr_auc - classifier_metrics.pr_auc,
            "encoder_beats_classifier": encoder_metrics.f1 > classifier_metrics.f1,
        },
    }

    if sbert_model_name is not None:
        scorer = SbertPretrainedPairScorer(sbert_model_name)
        if sbert_model_factory is not None:
            scorer.model_factory = sbert_model_factory
        sbert_start = time.perf_counter()
        sbert_results = cross_train_pairs(split, {"sbert_pretrained": scorer})
        fit_seconds["sbert_pretrained_baseline"] = time.perf_counter() - sbert_start
        sbert_metrics = sbert_results["sbert_pretrained"].test_metrics

        report["sbert_baseline"] = {
            "name": "sbert_pretrained",
            "model_name": sbert_model_name,
            "resolved_checkpoint": resolve_sbert_model_name(sbert_model_name),
            "validation_metrics": asdict(
                sbert_results["sbert_pretrained"].validation_metrics
            ),
            "test_metrics": asdict(sbert_metrics),
        }
        report["ceiling"]["sbert_f1"] = sbert_metrics.f1
        report["ceiling"]["encoder_vs_sbert_f1_delta"] = (
            encoder_metrics.f1 - sbert_metrics.f1
        )
        report["ceiling"]["sbert_pr_auc"] = sbert_metrics.pr_auc
        report["ceiling"]["encoder_vs_sbert_pr_auc_delta"] = (
            encoder_metrics.pr_auc - sbert_metrics.pr_auc
        )
        report["ceiling"]["encoder_beats_sbert"] = encoder_metrics.f1 > sbert_metrics.f1

    return report


_DATE_SETTINGS = (
    RoleSetting(
        "date",
        "source",
        f"The canonical snapshot read (default: {_CANONICAL_DATE}).",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_roots_args(parser)
    parser.add_argument("--system", default=_SYSTEM)
    add_role_settings(parser, _DATE_SETTINGS)
    parser.add_argument("--jurisdiction", default=_JURISDICTION)
    add_declared_arguments(parser, DECLARATIONS, SETTING_SURFACE)
    parser.add_argument(
        "--corpus-out",
        default=None,
        help="Where to write the WordPiece training corpus (never under data/). "
        "Defaults under --temp-dir; a relative path resolves against the checkout.",
    )
    parser.add_argument(
        "--tokenizer-out",
        default=None,
        help="Where to write the trained WordPiece tokenizer artifact. Defaults "
        "under --temp-dir; a relative path resolves against the checkout.",
    )
    parser.add_argument(
        "--skip-sbert",
        action="store_true",
        help=(
            "Omit the pretrained-S-BERT baseline. Use when the optional "
            "'sentence-transformers' dependency (company_vectorize's 'sbert' extra) is not "
            "synced into this .venv."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Where to write the full measurement report. A relative path resolves "
        "against the checkout.",
    )
    add_dry_run_arg(parser)
    return parser


def _resolve_corpus_out(roots: WorkspaceRoots, value: str | None) -> Path:
    return resolve_relative_to_checkout(roots, value) or (
        roots.temp
        / "pooled_subword_encoder_ceiling"
        / "wordpiece_training_corpus.parquet"
    )


def _resolve_tokenizer_out(roots: WorkspaceRoots, value: str | None) -> Path:
    return resolve_relative_to_checkout(roots, value) or (
        roots.temp / "pooled_subword_encoder_ceiling" / "wordpiece_tokenizer.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.canonical_date = (
        resolve_role_settings(args, _DATE_SETTINGS)["source_date"] or _CANONICAL_DATE
    )
    roots = resolve_workspace_roots_from_args(args)

    resolved = resolve_declared_settings(args, DECLARATIONS, SETTING_SURFACE)
    values = resolved_setting_values(resolved)
    corpus_out = _resolve_corpus_out(roots, args.corpus_out)
    tokenizer_out = _resolve_tokenizer_out(roots, args.tokenizer_out)
    json_out = resolve_relative_to_checkout(roots, args.json_out)
    sbert_model_name = None if args.skip_sbert else values["sbert_model_name"]

    if args.dry_run:
        report_resolved_settings(
            "measure_pooled_subword_encoder_ceiling",
            resolved,
            system=args.system,
            jurisdiction=args.jurisdiction,
            canonical_date=args.canonical_date,
            skip_sbert=args.skip_sbert,
            corpus_out=corpus_out,
            tokenizer_out=tokenizer_out,
            json_out=json_out,
        )
        outputs = [
            PlannedOutput(corpus_out, OUTPUT_CLEAR, note="WordPiece training corpus"),
            PlannedOutput(
                tokenizer_out, OUTPUT_CLEAR, note="trained WordPiece tokenizer"
            ),
        ]
        if json_out is not None:
            outputs.append(
                PlannedOutput(json_out, OUTPUT_CLEAR, note="full measurement report")
            )
        report_output_plan("[dry-run]  ", outputs)
        if json_out is None:
            print("[dry-run]   --json-out not given; report would not be written")
        return 0

    variants = load_real_alias_variants(
        roots,
        system=args.system,
        canonical_date=args.canonical_date,
        jurisdiction_code=args.jurisdiction,
    )
    print(
        f"{len(variants)} real name-variant rows loaded for "
        f"{args.system}/{args.jurisdiction} ({args.canonical_date})"
    )

    tokenizer_path = train_tokenizer_over_names(
        [variant.name for variant in variants],
        corpus_out=corpus_out,
        tokenizer_out=tokenizer_out,
    )

    splitter, negatives = resolve_splitter_and_negatives(values)

    report = build_ceiling_report(
        system=args.system,
        jurisdiction_code=args.jurisdiction,
        canonical_date=args.canonical_date,
        splitter=splitter,
        negatives=negatives,
        variants=variants,
        tokenizer_path=tokenizer_path,
        sbert_model_name=sbert_model_name,
    )

    print(json.dumps(report["run"], indent=2))
    print(json.dumps(report["ceiling"], indent=2))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote full measurement to {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
