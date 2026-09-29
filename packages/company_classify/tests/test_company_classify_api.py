import pytest
from company_classify import (
    LabeledExample,
    TfidfLogRegClassifier,
    TfidfMlpClassifier,
    cross_train,
    predict_batch,
    split_examples,
)
from company_classify.models import build_char_ngram_vectorizer


def test_the_shared_vectorizer_counts_lowercased_word_bounded_character_ngrams() -> (
    None
):
    vectorizer = build_char_ngram_vectorizer(ngram_min=2, ngram_max=3, max_features=50)

    vectorizer.fit(["Ab C"])

    assert vectorizer.max_features == 50
    assert set(vectorizer.vocabulary_) == {
        " a",
        "ab",
        "b ",
        " c",
        "c ",
        " ab",
        "ab ",
        " c ",
    }


def _sample_examples() -> list[LabeledExample]:
    positives = [
        "Acme Holdings Ltd",
        "Blue River Technologies GmbH",
        "Northshore Trading LLC",
        "Westfield Logistics Limited",
        "Meridian Group PLC",
        "Summit Capital Partners",
    ]
    negatives = [
        "hello world",
        "random sentence only",
        "totally unrelated note",
        "just numbers 12345",
        "colourless green ideas",
        "small text sample",
    ]

    return [
        *[LabeledExample(text=text, label=1) for text in positives],
        *[LabeledExample(text=text, label=0) for text in negatives],
    ]


def test_split_examples_is_deterministic_for_seed():
    examples = _sample_examples()
    a = split_examples(examples, seed=11)
    b = split_examples(examples, seed=11)

    assert [item.text for item in a.train] == [item.text for item in b.train]
    assert [item.text for item in a.validation] == [item.text for item in b.validation]
    assert [item.text for item in a.test] == [item.text for item in b.test]


@pytest.mark.integration
def test_cross_train_returns_metrics_for_both_models():
    split = split_examples(_sample_examples(), seed=7)

    results = cross_train(
        split,
        {
            "logreg": TfidfLogRegClassifier(max_features=512),
            "mlp": TfidfMlpClassifier(
                max_features=512, hidden_layer_sizes=(16,), random_state=7
            ),
        },
    )

    assert set(results.keys()) == {"logreg", "mlp"}

    for result in results.values():
        assert 0.0 <= result.validation_metrics.f1 <= 1.0
        assert 0.0 <= result.test_metrics.pr_auc <= 1.0
        assert 0.0 <= result.validation_metrics.recall_at_k <= 1.0
        assert 0.0 <= result.test_metrics.recall_at_k <= 1.0
        assert 0.0 <= result.validation_metrics.candidate_set_size_ratio <= 1.0
        assert 0.0 <= result.test_metrics.candidate_set_size_ratio <= 1.0
        assert 0.0 <= result.validation_metrics.precision_at_k <= 1.0
        assert 0.0 <= result.test_metrics.precision_at_k <= 1.0
        assert 0.0 <= result.validation_metrics.duplication_rate <= 1.0
        assert 0.0 <= result.test_metrics.duplication_rate <= 1.0
        assert result.validation_metrics.latency_per_1k_rows >= 0.0
        assert result.test_metrics.latency_per_1k_rows >= 0.0


def test_predict_batch_returns_score_and_label():
    split = split_examples(_sample_examples(), seed=4)
    model = TfidfLogRegClassifier(max_features=256)
    model.fit(split.train)

    rows = predict_batch(
        model, ["Acme Holdings Ltd", "just random words"], threshold=0.5
    )

    assert len(rows) == 2
    assert rows[0].text == "Acme Holdings Ltd"
    assert 0.0 <= rows[0].score <= 1.0
    assert rows[0].label in {0, 1}
