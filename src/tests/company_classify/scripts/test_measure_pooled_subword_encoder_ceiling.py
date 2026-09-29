from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from company_classify import EntitySplitter, NameVariant, NegativeSamplingConfig
from company_tokenize import load_tokenizer_vocabulary

from scripts.measure_pooled_subword_encoder_ceiling import (
    SbertPretrainedPairScorer,
    build_ceiling_report,
    main,
    train_tokenizer_over_names,
)

_BASE_NAMES = [
    "acme systems",
    "blue river technologies",
    "northshore trading",
    "westfield logistics",
    "meridian group",
    "summit capital partners",
    "cedar valley holdings",
    "ironbridge freight",
    "lighthouse consulting",
    "harbor point ventures",
    "silverline manufacturing",
    "brightwater energy",
    "stonegate insurance",
    "pinecrest logistics",
    "oakridge financial",
    "riverside analytics",
    "crestview partners",
    "amberfield industries",
    "goldenrod capital",
    "eastbrook holdings",
]
_LEGAL_SUFFIXES = ["limited", "ltd", "gmbh", "llc", "plc", "inc"]


def _variants(count: int = 40) -> list[NameVariant]:
    variants: list[NameVariant] = []
    for index in range(count):
        base = _BASE_NAMES[index % len(_BASE_NAMES)]
        uri = f"wikidata://Q{index:06d}"
        primary = f"{base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}"
        previous = f"{base} {_LEGAL_SUFFIXES[(index + 1) % len(_LEGAL_SUFFIXES)]}"
        variants.append(NameVariant(uri, primary, "primary"))
        variants.append(NameVariant(uri, previous, "previous"))
    return variants


class _FakeSentenceTransformer:
    """A fake model with the `sentence_transformers.SentenceTransformer.encode()` shape.

    Assigns each distinct word a fixed random direction and sums per-name, so two names sharing
    words score more similar than two that don't -- enough structure for `predict_scores()`'s
    arithmetic to be worth checking, without ever importing `sentence_transformers`.
    """

    def __init__(self) -> None:
        self._rng = np.random.default_rng(7)
        self._word_vectors: dict[str, np.ndarray] = {}

    def _word_vector(self, word: str) -> np.ndarray:
        vector = self._word_vectors.get(word)
        if vector is None:
            vector = self._rng.normal(size=8)
            self._word_vectors[word] = vector
        return vector

    def encode(self, names: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = []
        for name in names:
            summed = sum(
                (self._word_vector(word) for word in name.split()),
                start=np.zeros(8),
            )
            if normalize_embeddings:
                norm = np.linalg.norm(summed)
                summed = summed / norm if norm > 0 else summed
            vectors.append(summed)
        return np.array(vectors)


def test_sbert_pretrained_pair_scorer_fit_is_a_noop_and_scores_via_injected_model() -> (
    None
):
    scorer = SbertPretrainedPairScorer(
        model_name="fake-checkpoint",
        model_factory=lambda checkpoint: _FakeSentenceTransformer(),
    )

    scorer.fit(
        []
    )  # pretrained: no training step, must not raise or mutate anything observable

    scores = scorer.predict_scores(
        [
            ("acme systems limited", "acme systems limited"),
            ("acme systems limited", "westfield logistics ltd"),
        ]
    )

    assert len(scores) == 2
    for score in scores:
        assert 0.0 <= score <= 1.0
    # Identical names must score at least as high as two names sharing no words.
    assert scores[0] >= scores[1]


def test_sbert_pretrained_pair_scorer_caches_embeddings_across_calls() -> None:
    calls: list[list[str]] = []

    class _CountingFakeModel(_FakeSentenceTransformer):
        def encode(
            self, names: list[str], normalize_embeddings: bool = True
        ) -> np.ndarray:
            calls.append(list(names))
            return super().encode(names, normalize_embeddings=normalize_embeddings)

    fake_model = _CountingFakeModel()
    scorer = SbertPretrainedPairScorer(
        model_name="fake-checkpoint", model_factory=lambda checkpoint: fake_model
    )

    scorer.predict_scores([("acme systems", "acme systems")])
    scorer.predict_scores([("acme systems", "acme systems")])

    # The second call's name was already cached, so only the first call should have reached
    # the (fake) model's encode().
    assert len(calls) == 1


@pytest.mark.integration
def test_build_ceiling_report_includes_sbert_baseline_via_injected_factory(
    tmp_path,
) -> None:
    splitter = EntitySplitter(seed=42)
    negatives = NegativeSamplingConfig(seed=42)
    variants = _variants()

    corpus_out = tmp_path / "wordpiece_training_corpus.parquet"
    tokenizer_out = tmp_path / "wordpiece_tokenizer.json"
    tokenizer_path = train_tokenizer_over_names(
        [variant.name for variant in variants],
        corpus_out=corpus_out,
        tokenizer_out=tokenizer_out,
    )
    load_tokenizer_vocabulary(
        tokenizer_path, trainer="wordpiece"
    )  # sanity: artifact loads

    report = build_ceiling_report(
        system="wikidata",
        jurisdiction_code="GB",
        canonical_date="2026-07-16",
        splitter=splitter,
        negatives=negatives,
        variants=variants,
        tokenizer_path=tokenizer_path,
        sbert_model_name="fake-checkpoint",
        sbert_model_factory=lambda checkpoint: _FakeSentenceTransformer(),
    )

    assert report["sbert_baseline"]["name"] == "sbert_pretrained"
    assert report["sbert_baseline"]["model_name"] == "fake-checkpoint"
    metrics = report["sbert_baseline"]["test_metrics"]
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    ceiling = report["ceiling"]
    assert ceiling["sbert_f1"] == pytest.approx(metrics["f1"])
    assert ceiling["encoder_vs_sbert_f1_delta"] == pytest.approx(
        ceiling["encoder_f1"] - ceiling["sbert_f1"]
    )
    assert isinstance(ceiling["encoder_beats_sbert"], bool)


@pytest.mark.integration
def test_build_ceiling_report_skips_sbert_baseline_when_model_name_is_none(
    tmp_path,
) -> None:
    splitter = EntitySplitter(seed=42)
    negatives = NegativeSamplingConfig(seed=42)
    variants = _variants()

    corpus_out = tmp_path / "wordpiece_training_corpus.parquet"
    tokenizer_out = tmp_path / "wordpiece_tokenizer.json"
    tokenizer_path = train_tokenizer_over_names(
        [variant.name for variant in variants],
        corpus_out=corpus_out,
        tokenizer_out=tokenizer_out,
    )

    report = build_ceiling_report(
        system="wikidata",
        jurisdiction_code="GB",
        canonical_date="2026-07-16",
        splitter=splitter,
        negatives=negatives,
        variants=variants,
        tokenizer_path=tokenizer_path,
        sbert_model_name=None,
    )

    assert "sbert_baseline" not in report
    assert "sbert_f1" not in report["ceiling"]


def test_dry_run_reports_the_plan_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    temp_dir = tmp_path / "temp"
    corpus_out = tmp_path / "reports" / "corpus.parquet"
    tokenizer_out = tmp_path / "reports" / "tokenizer.json"
    json_out = tmp_path / "reports" / "encoder_ceiling.json"

    exit_code = main(
        [
            "--dry-run",
            "--data-dir",
            str(data_dir),
            "--temp-dir",
            str(temp_dir),
            "--split-seed",
            "7",
            "--corpus-out",
            str(corpus_out),
            "--tokenizer-out",
            str(tokenizer_out),
            "--json-out",
            str(json_out),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "split_seed=7 (given)" in out
    assert (
        "sbert_model_name='sentence-transformers/all-MiniLM-L6-v2' (script default)"
        in out
    )
    assert "skip_sbert=False" in out
    assert f"would clear {corpus_out}" in out
    assert f"would clear {tokenizer_out}" in out
    assert f"would clear {json_out}" in out

    assert not corpus_out.exists()
    assert not tokenizer_out.exists()
    assert not json_out.exists()
    assert not data_dir.exists()
    assert not temp_dir.exists()


def test_dry_run_defaults_corpus_and_tokenizer_output_under_the_temp_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    temp_dir = tmp_path / "temp"

    exit_code = main(["--dry-run", "--temp-dir", str(temp_dir), "--skip-sbert"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert (
        temp_dir
        / "pooled_subword_encoder_ceiling"
        / "wordpiece_training_corpus.parquet"
    ).as_posix() in out
    assert (
        temp_dir / "pooled_subword_encoder_ceiling" / "wordpiece_tokenizer.json"
    ).as_posix() in out
    assert "skip_sbert=True" in out
    assert "json_out=None" in out
    assert "report would not be written" in out
