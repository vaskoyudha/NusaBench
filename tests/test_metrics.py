from __future__ import annotations

# pyright: reportUnknownMemberType=false
import pytest

from nusabench.metrics import MetricRegistry, compute_metrics
from nusabench.metrics.accuracy import AccuracyMetric
from nusabench.metrics.bleu import BleuMetric
from nusabench.metrics.chrf import ChrFMetric
from nusabench.metrics.exact_match import ExactMatchMetric
from nusabench.metrics.f1 import F1Metric
from nusabench.metrics.rouge import RougeMetric


def test_accuracy_metric_computes_expected_value() -> None:
    result = AccuracyMetric().compute([0, 1, 1], [0, 1, 0])

    assert result == {"accuracy": pytest.approx(2 / 3)}


def test_f1_metric_returns_value_between_zero_and_one() -> None:
    result = F1Metric().compute([0, 1, 1], [0, 1, 0])

    assert "f1" in result
    assert 0.0 <= result["f1"] <= 1.0


def test_rouge_metric_returns_perfect_score_for_identical_text() -> None:
    result = RougeMetric().compute(["hello world"], ["hello world"])

    assert result["rouge1"] == pytest.approx(1.0)


def test_bleu_metric_is_high_for_identical_text() -> None:
    result = BleuMetric().compute(["ini bagus"], ["ini bagus"])

    assert result["bleu"] > 90.0


def test_chrf_metric_is_high_for_identical_text() -> None:
    result = ChrFMetric().compute(["ini bagus"], ["ini bagus"])

    assert result["chrf"] > 90.0


def test_exact_match_metric_returns_one_for_exact_match() -> None:
    result = ExactMatchMetric().compute(["Jakarta"], ["Jakarta"])

    assert result == {"exact_match": 1.0}


def test_exact_match_metric_handles_empty_lists() -> None:
    result = ExactMatchMetric().compute([], [])

    assert result == {"exact_match": 0.0}


def test_metric_registry_returns_accuracy_metric() -> None:
    assert MetricRegistry.get("accuracy") is AccuracyMetric


def test_metric_registry_lists_all_metrics() -> None:
    assert MetricRegistry.list() == ["accuracy", "bleu", "chrf", "exact_match", "f1", "rouge"]


def test_compute_metrics_returns_aggregated_results() -> None:
    result = compute_metrics(["accuracy"], [1, 0], [1, 1])

    assert result == {"accuracy": pytest.approx(0.5)}
