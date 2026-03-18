from __future__ import annotations

from collections.abc import Callable

from .base import Metric


class MetricRegistry:
    _registry: dict[str, type[Metric]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Metric]], type[Metric]]:
        def decorator(metric_cls: type[Metric]) -> type[Metric]:
            cls._registry[name] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Metric]:
        if name not in cls._registry:
            raise KeyError(f"Metric '{name}' not found. Available: {cls.list()}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry)


def compute_metrics(
    metric_names: list[str],
    predictions: list[object],
    references: list[object],
) -> dict[str, float]:
    results: dict[str, float] = {}
    for metric_name in metric_names:
        metric_class = MetricRegistry.get(metric_name)
        metric = metric_class()
        results.update(metric.compute(predictions=predictions, references=references))
    return results


from .accuracy import AccuracyMetric  # noqa: F401, E402
from .bleu import BleuMetric  # noqa: F401, E402
from .chrf import ChrFMetric  # noqa: F401, E402
from .exact_match import ExactMatchMetric  # noqa: F401, E402
from .f1 import F1Metric  # noqa: F401, E402
from .rouge import RougeMetric  # noqa: F401, E402

_ = MetricRegistry.register("accuracy")(AccuracyMetric)
_ = MetricRegistry.register("bleu")(BleuMetric)
_ = MetricRegistry.register("chrf")(ChrFMetric)
_ = MetricRegistry.register("exact_match")(ExactMatchMetric)
_ = MetricRegistry.register("f1")(F1Metric)
_ = MetricRegistry.register("rouge")(RougeMetric)

__all__ = [
    "AccuracyMetric",
    "BleuMetric",
    "ChrFMetric",
    "ExactMatchMetric",
    "F1Metric",
    "Metric",
    "MetricRegistry",
    "RougeMetric",
    "compute_metrics",
]
