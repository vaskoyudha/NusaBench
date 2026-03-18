from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from nusabench.utils.config import NusaBenchConfig
from nusabench.utils.data import format_prompt_jinja, load_hf_dataset
from nusabench.utils.logging import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)


def test_get_logger_idempotent():
    a = get_logger("test")
    b = get_logger("test")
    assert a is b


def test_config_defaults_and_custom():
    cfg = NusaBenchConfig()
    assert cfg.seed == 42
    assert cfg.verbose is False
    assert cfg.limit is None

    cfg2 = NusaBenchConfig(seed=99, verbose=True, limit=10)
    assert cfg2.seed == 99
    assert cfg2.verbose is True
    assert cfg2.limit == 10


def test_format_prompt_jinja_basic():
    assert format_prompt_jinja("Hello {{name}}", {"name": "World"}) == "Hello World"


def test_format_prompt_jinja_indonesian():
    assert (
        format_prompt_jinja("Analisis sentimen: {{text}}", {"text": "Makanan ini enak"})
        == "Analisis sentimen: Makanan ini enak"
    )


def test_format_prompt_jinja_concat():
    assert format_prompt_jinja("{{a}} + {{b}}", {"a": "satu", "b": "dua"}) == "satu + dua"


def test_load_hf_dataset_calls_load_dataset():
    mock_dataset = [{"text": "hello", "label": 0}]
    with patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        result = load_hf_dataset("fake/dataset", split="test")
        mock_load.assert_called_once_with("fake/dataset", None, split="test", cache_dir=None)
        assert result == [{"text": "hello", "label": 0}]


def test_load_hf_dataset_limit():
    mock_dataset = [{"text": f"ex{i}", "label": i} for i in range(10)]
    with patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        result = load_hf_dataset("fake/dataset", limit=3)
        assert len(result) == 3


def test_load_hf_dataset_error():
    with patch("datasets.load_dataset", side_effect=Exception("not found")), pytest.raises(
        RuntimeError, match="Failed to load dataset"
    ):
        load_hf_dataset("bad/dataset")
