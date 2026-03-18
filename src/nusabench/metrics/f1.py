from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportImplicitOverride=false, reportInvalidCast=false
import os
from functools import lru_cache
from typing import Protocol, cast

import evaluate  # type: ignore[import-untyped]

from .base import Metric

EVALUATE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "nusabench", "evaluate")


class _F1Evaluator(Protocol):
    def compute(
        self,
        *,
        predictions: list[object],
        references: list[object],
        average: str,
    ) -> dict[str, float]: ...


@lru_cache(maxsize=1)
def _load_f1_metric() -> _F1Evaluator:
    try:
        return cast(_F1Evaluator, cast(object, evaluate.load("f1", cache_dir=EVALUATE_CACHE_DIR)))
    except Exception:
        return cast(_F1Evaluator, cast(object, evaluate.load("f1")))


class F1Metric(Metric):
    @property
    def name(self) -> str:
        return "f1"

    def compute(self, predictions: list[object], references: list[object]) -> dict[str, float]:
        result = _load_f1_metric().compute(
            predictions=predictions,
            references=references,
            average="weighted",
        )
        return {"f1": float(result["f1"])}
