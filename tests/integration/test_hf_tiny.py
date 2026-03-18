from __future__ import annotations

from unittest.mock import patch

import pytest

from nusabench import EvaluationResult, evaluate

SENTIMENT_DATASET: list[dict[str, object]] = [
    {"text": "Bagus sekali!", "label": "positif"},
    {"text": "Buruk sekali!", "label": "negatif"},
]


@pytest.mark.slow
def test_hf_tiny_gpt2_sentiment() -> None:
    with patch("nusabench.tasks.base.Task.load_dataset", return_value=SENTIMENT_DATASET):
        result = evaluate(
            model="hf",
            tasks=["sentiment_smsa"],
            model_args="pretrained=sshleifer/tiny-gpt2",
            limit=2,
        )

    assert isinstance(result, EvaluationResult)
    assert "sentiment_smsa" in result.results
    assert result.results["sentiment_smsa"].num_samples == 2
