import json
from pathlib import Path

from company_classify import (
    LabeledExample,
    TfidfLogRegClassifier,
    TfidfPairMlpClassifier,
    cross_train,
    save_comparison_report_json,
    save_run_summary_json,
    split_examples,
)
from company_classify.comparison import ComparisonReport
from company_classify.pair_train import TrainedPairModelResult
from company_classify.schema import MetricBundle


def test_save_run_summary_json_writes_metric_payload(tmp_path: Path):
    examples = [
        LabeledExample(text="Acme Ltd", label=1),
        LabeledExample(text="Blue River GmbH", label=1),
        LabeledExample(text="free text line", label=0),
        LabeledExample(text="another plain sentence", label=0),
        LabeledExample(text="Northshore Trading LLC", label=1),
        LabeledExample(text="not a company", label=0),
    ]
    split = split_examples(examples, seed=2)

    results = cross_train(split, {"logreg": TfidfLogRegClassifier(max_features=128)})

    path = tmp_path / "run" / "summary.json"
    save_run_summary_json(results, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "logreg" in payload
    assert "validation_metrics" in payload["logreg"]
    assert "test_metrics" in payload["logreg"]
    assert "recall_at_k" in payload["logreg"]["validation_metrics"]
    assert "candidate_set_size_ratio" in payload["logreg"]["validation_metrics"]
    assert "precision_at_k" in payload["logreg"]["validation_metrics"]
    assert "duplication_rate" in payload["logreg"]["validation_metrics"]
    assert "latency_per_1k_rows" in payload["logreg"]["validation_metrics"]


def _fake_pair_result(f1: float) -> TrainedPairModelResult:
    metrics = MetricBundle(
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
    return TrainedPairModelResult(
        name="fake",
        model=TfidfPairMlpClassifier(),
        validation_metrics=metrics,
        test_metrics=metrics,
    )


def test_save_comparison_report_json_writes_a_grouped_payload(tmp_path: Path):
    report = ComparisonReport(
        supervised={"pair_mlp": _fake_pair_result(0.6)},
        self_supervised={"pooled_subword": _fake_pair_result(0.4)},
    )

    path = tmp_path / "run" / "comparison.json"
    save_comparison_report_json(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"supervised", "self_supervised"}
    assert payload["supervised"]["pair_mlp"]["test_metrics"]["f1"] == 0.6
    assert payload["self_supervised"]["pooled_subword"]["test_metrics"]["f1"] == 0.4
