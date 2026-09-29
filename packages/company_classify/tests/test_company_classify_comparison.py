from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import pytest
from company_classify import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    ComparisonReport,
    EntitySplitter,
    ModelCategory,
    PairRecord,
    PairSource,
    PooledSubwordContrastiveEncoder,
    RepeatabilityResult,
    TfidfPairMlpClassifier,
    check_repeatability,
    compare_pair_models,
    split_pairs,
)
from company_classify.pair_train import TrainedPairModelResult
from company_classify.schema import MetricBundle

SPLITTER = EntitySplitter(seed=42)

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
]
_LEGAL_SUFFIXES = ["limited", "ltd", "gmbh", "llc", "plc", "inc"]

UNK_TOKEN = "[UNK]"


def _whitespace_vocab() -> dict[str, int]:
    """A word-level "subword" vocabulary, real enough to pool/train over without a real
    WordPiece/SentencePiece artifact, which this direct test has no business training."""
    words = {word for name in _BASE_NAMES for word in name.split()}
    words.update(_LEGAL_SUFFIXES)
    vocab = {word: index for index, word in enumerate(sorted(words))}
    vocab[UNK_TOKEN] = len(vocab)
    return vocab


def _encoder(random_state: int = 3) -> PooledSubwordContrastiveEncoder:
    return PooledSubwordContrastiveEncoder(
        vocab=_whitespace_vocab(),
        encode=str.split,
        unk_token=UNK_TOKEN,
        embed_dim=8,
        epochs=15,
        random_state=random_state,
    )


def _pairs() -> list[PairRecord]:
    """Positives: a base name against a different legal-suffix variant of itself. Negatives:
    that same base name against a different company's name entirely."""
    pairs: list[PairRecord] = []
    for index, base in enumerate(_BASE_NAMES):
        uri = f"co://{index:04d}"
        other_uri = f"co://{(index + 1) % len(_BASE_NAMES):04d}"
        other_base = _BASE_NAMES[(index + 1) % len(_BASE_NAMES)]
        anchor = f"{base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}"
        variant = f"{base} {_LEGAL_SUFFIXES[(index + 1) % len(_LEGAL_SUFFIXES)]}"
        split = SPLITTER.assign(uri)

        pairs.append(
            PairRecord(
                left_name=anchor,
                right_name=variant,
                label=MATCH_LABEL,
                source=PairSource.SYNTHETIC,
                split=split,
                left_system_uri=uri,
                right_system_uri=uri,
            )
        )
        pairs.append(
            PairRecord(
                left_name=anchor,
                right_name=f"{other_base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}",
                label=NON_MATCH_LABEL,
                source=PairSource.SYNTHETIC,
                split=split,
                left_system_uri=uri,
                right_system_uri=other_uri,
            )
        )
    return pairs


def _metrics(f1: float) -> MetricBundle:
    return MetricBundle(
        precision=f1,
        recall=f1,
        f1=f1,
        pr_auc=f1,
        recall_at_k=f1,
        candidate_set_size_ratio=f1,
        precision_at_k=f1,
        duplication_rate=f1,
        latency_per_1k_rows=0.0,
    )


def _fake_result(f1: float) -> TrainedPairModelResult:
    return TrainedPairModelResult(
        name="fake",
        model=TfidfPairMlpClassifier(),
        validation_metrics=_metrics(f1),
        test_metrics=_metrics(f1),
    )


@dataclass
class _NondeterministicScorer:
    """A `PairClassifierModel` whose scores depend on which instance is asked: each new one
    ranks the pairs in the opposite order to the last, so two independent fits over the same
    pairs are certain to land on different `MetricBundle`s -- the negative case
    `check_repeatability()` exists to surface. Unseeded random scores left the two bundles
    equal about one run in eight, since so small a split quantises every metric to a handful
    of values."""

    _instances: ClassVar[int] = 0

    def __post_init__(self) -> None:
        type(self)._instances += 1
        self._descending = type(self)._instances % 2 == 0

    def fit(self, pairs: Sequence[PairRecord]) -> None:
        return None

    def predict_scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = [(index + 1) / (len(pairs) + 1) for index in range(len(pairs))]
        return scores[::-1] if self._descending else scores


@pytest.mark.integration
def test_compare_pair_models_groups_results_by_category() -> None:
    split = split_pairs(_pairs())

    report = compare_pair_models(
        split,
        supervised={
            "pair_mlp": TfidfPairMlpClassifier(
                max_features=256, hidden_layer_sizes=(8,)
            )
        },
        self_supervised={"pooled_subword": _encoder()},
    )

    assert set(report.supervised.keys()) == {"pair_mlp"}
    assert set(report.self_supervised.keys()) == {"pooled_subword"}
    for pool in (report.supervised, report.self_supervised):
        for result in pool.values():
            assert 0.0 <= result.test_metrics.f1 <= 1.0


def test_compare_pair_models_rejects_empty_supervised() -> None:
    split = split_pairs(_pairs())

    with pytest.raises(ValueError):
        compare_pair_models(split, supervised={}, self_supervised={"enc": _encoder()})


def test_compare_pair_models_rejects_empty_self_supervised() -> None:
    split = split_pairs(_pairs())

    with pytest.raises(ValueError):
        compare_pair_models(
            split,
            supervised={"mlp": TfidfPairMlpClassifier()},
            self_supervised={},
        )


def test_comparison_report_best_returns_the_highest_scoring_model() -> None:
    report = ComparisonReport(
        supervised={"weak": _fake_result(0.2), "strong": _fake_result(0.8)},
        self_supervised={"only": _fake_result(0.5)},
    )

    name, result = report.best(ModelCategory.SUPERVISED)

    assert name == "strong"
    assert result.test_metrics.f1 == 0.8


def test_comparison_report_best_raises_for_an_empty_category() -> None:
    report = ComparisonReport(
        supervised={}, self_supervised={"only": _fake_result(0.5)}
    )

    with pytest.raises(ValueError):
        report.best(ModelCategory.SUPERVISED)


def test_comparison_report_delta_is_self_supervised_minus_supervised() -> None:
    report = ComparisonReport(
        supervised={"sup": _fake_result(0.4)},
        self_supervised={"self": _fake_result(0.7)},
    )

    assert report.delta() == pytest.approx(0.3)


@pytest.mark.integration
def test_check_repeatability_flags_a_seeded_model_as_reproducible() -> None:
    split = split_pairs(_pairs())

    results = check_repeatability(
        split,
        {"pooled_subword": lambda: _encoder(random_state=7)},
        tolerance=1e-9,
    )

    result = results["pooled_subword"]
    assert isinstance(result, RepeatabilityResult)
    assert result.reproducible
    assert result.max_abs_diff == pytest.approx(0.0, abs=1e-9)


def test_check_repeatability_flags_an_unseeded_model_as_not_reproducible() -> None:
    split = split_pairs(_pairs())

    results = check_repeatability(
        split, {"random_scorer": _NondeterministicScorer}, tolerance=1e-9
    )

    result = results["random_scorer"]
    assert result.name == "random_scorer"
    assert not result.reproducible
    assert result.max_abs_diff > 1e-9


def test_check_repeatability_rejects_empty_model_factories() -> None:
    split = split_pairs(_pairs())

    with pytest.raises(ValueError):
        check_repeatability(split, {})
