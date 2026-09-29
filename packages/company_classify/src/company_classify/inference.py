"""Score new texts with a fitted `ClassifierModel`, returning each text's score and thresholded label."""

from __future__ import annotations

from .models import ClassifierModel
from .schema import Prediction


def predict_batch(
    model: ClassifierModel, texts: list[str], threshold: float = 0.5
) -> list[Prediction]:
    scores = model.predict_scores(texts)
    return [
        Prediction(
            text=text,
            score=float(score),
            label=1 if float(score) >= threshold else 0,
        )
        for text, score in zip(texts, scores, strict=True)
    ]
