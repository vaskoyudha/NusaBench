from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportImplicitOverride=false
from typing import Protocol, cast

import sacrebleu  # type: ignore[import-untyped]

from .base import Metric


class _ScoreResult(Protocol):
    score: float


class ChrFMetric(Metric):
    @property
    def name(self) -> str:
        return "chrf"

    def compute(self, predictions: list[object], references: list[object]) -> dict[str, float]:
        if not predictions or not references:
            return {"chrf": 0.0}

        hypotheses = [str(prediction) for prediction in predictions]
        wrapped_references = [[str(reference) for reference in references]]
        result = cast(
            _ScoreResult,
            sacrebleu.corpus_chrf(hypotheses, wrapped_references),
        )
        return {"chrf": float(result.score)}
