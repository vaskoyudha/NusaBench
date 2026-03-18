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


def _are_integer_labels(values: list[object]) -> bool:
    """Check if all values are integers or string representations of integers."""
    for v in values:
        s = str(v).strip()
        if not (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
            return False
    return True


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between two strings (word overlap).

    This is the standard QA F1 metric used in SQuAD-style evaluation.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = sum(1 for t in pred_tokens if t in ref_tokens)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class F1Metric(Metric):
    @property
    def name(self) -> str:
        return "f1"

    def compute(self, predictions: list[object], references: list[object]) -> dict[str, float]:
        if not predictions:
            return {"f1": 0.0}

        # If inputs look like integer labels, use the HuggingFace evaluate F1 metric
        if _are_integer_labels(predictions) and _are_integer_labels(references):
            int_preds = [int(str(p).strip()) for p in predictions]
            int_refs = [int(str(r).strip()) for r in references]
            result = _load_f1_metric().compute(
                predictions=int_preds,
                references=int_refs,
                average="weighted",
            )
            return {"f1": float(result["f1"])}

        # For string inputs (generate_until tasks), use token-level F1
        scores = [
            _token_f1(str(pred), str(ref))
            for pred, ref in zip(predictions, references, strict=False)
        ]
        return {"f1": sum(scores) / len(scores)}
