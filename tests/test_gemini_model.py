from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from nusabench.models import ModelRegistry
from nusabench.models.base import Model


def _make_gemini_model(
    api_key: str = "fake-key",
    model: str = "gemini-1.5-flash",
) -> Model:
    gemini_module = import_module("nusabench.models.gemini")
    cls = cast(type[Model], gemini_module.GeminiModel)

    fake_genai = MagicMock()
    with (
        patch.dict("sys.modules", {"google.generativeai": fake_genai, "google": MagicMock()}),
        patch("nusabench.models.gemini.import_module", return_value=fake_genai),
    ):
        instance = cls(model=model, api_key=api_key)
    return instance


def test_gemini_missing_api_key() -> None:
    gemini_module = import_module("nusabench.models.gemini")
    cls = gemini_module.GeminiModel

    with patch.dict("os.environ", {}, clear=True):
        env_without_key = {
            k: v for k, v in __import__("os").environ.items() if k != "GEMINI_API_KEY"
        }
        with (
            patch.dict("os.environ", env_without_key, clear=True),
            pytest.raises(ValueError, match="GEMINI_API_KEY"),
        ):
            cls()


def test_gemini_model_name() -> None:
    instance = _make_gemini_model()
    assert instance.model_name == "gemini:gemini-1.5-flash"


def test_gemini_supports_loglikelihood() -> None:
    instance = _make_gemini_model()
    assert instance.supports_loglikelihood() is False


def test_gemini_generate_mocked() -> None:
    gemini_module = import_module("nusabench.models.gemini")
    cls = cast(type[Model], gemini_module.GeminiModel)

    fake_genai = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "Ini adalah respons."
    fake_generative_model_instance = MagicMock()
    fake_generative_model_instance.generate_content.return_value = fake_response
    fake_genai.GenerativeModel.return_value = fake_generative_model_instance

    with patch("nusabench.models.gemini.import_module", return_value=fake_genai):
        instance = cls(model="gemini-1.5-flash", api_key="fake-key")

    instance._genai = fake_genai  # type: ignore[attr-defined]

    fake_api_core = MagicMock()
    with patch("nusabench.models.gemini.import_module", return_value=fake_api_core):
        results = instance.generate(["Halo dunia", "Apa kabar?"])

    assert results == ["Ini adalah respons.", "Ini adalah respons."]
    assert fake_genai.GenerativeModel.call_count == 2


def test_gemini_registry() -> None:
    gemini_module = import_module("nusabench.models.gemini")
    gemini_model_cls = gemini_module.GeminiModel

    assert ModelRegistry.get("gemini") is gemini_model_cls


def test_gemini_retry_on_rate_limit() -> None:
    gemini_module = import_module("nusabench.models.gemini")
    cls = cast(type[Model], gemini_module.GeminiModel)

    fake_genai = MagicMock()

    with patch("nusabench.models.gemini.import_module", return_value=fake_genai):
        instance = cls(model="gemini-1.5-flash", api_key="fake-key")

    instance._genai = fake_genai  # type: ignore[attr-defined]

    resource_exhausted = type("ResourceExhausted", (Exception,), {})

    fake_api_core_exceptions = SimpleNamespace(ResourceExhausted=resource_exhausted)

    success_response = MagicMock()
    success_response.text = "Success after retries"

    fake_model_instance = MagicMock()
    fake_model_instance.generate_content.side_effect = [
        resource_exhausted("rate limited"),
        resource_exhausted("rate limited"),
        success_response,
    ]
    fake_genai.GenerativeModel.return_value = fake_model_instance

    def import_side_effect(name: str) -> object:
        if name == "google.api_core.exceptions":
            return fake_api_core_exceptions
        return MagicMock()

    with (
        patch("nusabench.models.gemini.import_module", side_effect=import_side_effect),
        patch("nusabench.models.gemini.time.sleep") as mock_sleep,
    ):
        results = instance.generate(["Test prompt"])

    assert results == ["Success after retries"]
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)
