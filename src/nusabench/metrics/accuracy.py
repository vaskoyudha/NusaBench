from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportImplicitOverride=false, reportInvalidCast=false
import os
from functools import lru_cache
from typing import Protocol, cast

import evaluate  # type: ignore[import-untyped]

from .base import Metric

EVALUATE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "nusabench", "evaluate")


class _AccuracyEvaluator(Protocol):
    def compute(
        self,
        *,
        predictions: list[object],
        references: list[object],
    ) -> dict[str, float]: ...


@lru_cache(maxsize=1)
def _load_accuracy_metric() -> _AccuracyEvaluator:
    try:
        return cast(
            _AccuracyEvaluator,
            cast(object, evaluate.load("accuracy", cache_dir=EVALUATE_CACHE_DIR)),
        )
    except Exception:
        return cast(_AccuracyEvaluator, cast(object, evaluate.load("accuracy")))


class AccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, predictions: list[object], references: list[object]) -> dict[str, float]:
        result = _load_accuracy_metric().compute(predictions=predictions, references=references)
        return {"accuracy": float(result["accuracy"])}
