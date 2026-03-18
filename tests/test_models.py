from __future__ import annotations

import pytest
from typing_extensions import override

from nusabench.models import DummyModel, Model, ModelRegistry, register_model


def test_dummy_generate_returns_one_string_per_prompt() -> None:
    model = DummyModel()

    result = model.generate(["Hello", "World"])

    assert result == ["dummy response", "dummy response"]
    assert len(result) == 2
    assert all(isinstance(item, str) for item in result)


def test_dummy_loglikelihood_returns_one_float_per_prompt_pair() -> None:
    model = DummyModel()

    result = model.loglikelihood(["Hello"], ["World"])

    assert result == [-1.0]
    assert len(result) == 1
    assert all(isinstance(item, float) for item in result)


def test_dummy_supports_loglikelihood() -> None:
    assert DummyModel().supports_loglikelihood() is True


def test_model_registry_returns_dummy_model() -> None:
    assert ModelRegistry.get("dummy") is DummyModel


def test_model_registry_raises_for_unknown_model() -> None:
    with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
        _ = ModelRegistry.get("nonexistent")


def test_model_registry_lists_dummy() -> None:
    # Ensure the dummy model is registered. Other backends (e.g. 'hf')
    # may also be registered in CI, so check membership rather than exact list.
    assert "dummy" in ModelRegistry.list()


def test_register_model_decorator_registers_temp_class() -> None:
    @register_model("temp-test-model")
    class TempModel(Model):
        @override
        def generate(
            self,
            prompts: list[str],
            max_tokens: int = 256,
            **kwargs: object,
        ) -> list[str]:
            del max_tokens, kwargs
            return ["temp"] * len(prompts)

        @property
        @override
        def model_name(self) -> str:
            return "temp-test-model"

    assert ModelRegistry.get("temp-test-model") is TempModel
    assert "temp-test-model" in ModelRegistry.list()
