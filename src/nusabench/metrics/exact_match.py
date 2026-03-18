from __future__ import annotations

# pyright: reportImplicitOverride=false
from .base import Metric


class ExactMatchMetric(Metric):
    @property
    def name(self) -> str:
        return "exact_match"

    def compute(self, predictions: list[object], references: list[object]) -> dict[str, float]:
        if not predictions:
            return {"exact_match": 0.0}

        matches = sum(
            str(prediction).strip() == str(reference).strip()
            for prediction, reference in zip(predictions, references, strict=False)
        )
        return {"exact_match": matches / len(predictions)}
