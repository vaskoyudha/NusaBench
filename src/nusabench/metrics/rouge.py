from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportImplicitOverride=false
from typing import Protocol, cast

from rouge_score import rouge_scorer  # type: ignore[import-untyped]

from .base import Metric


class _RougeScore(Protocol):
    fmeasure: float


class RougeMetric(Metric):
    @property
    def name(self) -> str:
        return "rouge"

    def compute(self, predictions: list[object], references: list[object]) -> dict[str, float]:
        if not predictions or not references:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        pair_count = 0
        for prediction, reference in zip(predictions, references, strict=False):
            scores = cast(dict[str, _RougeScore], scorer.score(str(reference), str(prediction)))
            totals["rouge1"] += scores["rouge1"].fmeasure
            totals["rouge2"] += scores["rouge2"].fmeasure
            totals["rougeL"] += scores["rougeL"].fmeasure
            pair_count += 1

        if pair_count == 0:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        return {key: value / pair_count for key, value in totals.items()}
