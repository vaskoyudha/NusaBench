"""Tests for HuggingFace model backend."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from nusabench.models import Model, ModelRegistry

# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportMissingImports=false, reportUnknownLambdaType=false, reportUnknownVariableType=false, reportUnusedCallResult=false


def test_hf_model_registered() -> None:
    huggingface_module = import_module("nusabench.models.huggingface")
    huggingface_model = huggingface_module.HuggingFaceModel

    assert ModelRegistry.get("hf") is huggingface_model


def test_hf_model_registry_list_includes_hf() -> None:
    assert "hf" in ModelRegistry.list()


def test_parse_model_args() -> None:
    huggingface_module = import_module("nusabench.models.huggingface")
    parse_model_args = cast(
        Callable[[str], dict[str, str]],
        huggingface_module._parse_model_args,
    )

    assert parse_model_args("dtype=float16, device=cuda:0") == {
        "dtype": "float16",
        "device": "cuda:0",
    }
    assert parse_model_args("") == {}


def test_hf_model_raises_import_error_without_transformers() -> None:
    hf_model = cast(Callable[..., Model], ModelRegistry.get("hf"))

    real_import = __import__

    def fake_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name in {"torch", "transformers"}:
            raise ImportError("missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import), pytest.raises(
        ImportError, match="uv sync --extra hf"
    ):
        _ = hf_model(pretrained="fake/model")


def test_hf_model_init_generate_and_loglikelihood_with_mocks() -> None:
    hf_model_cls = cast(Callable[..., Model], ModelRegistry.get("hf"))

    fake_torch = SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )

    no_grad_context = MagicMock()
    no_grad_context.__enter__.return_value = None
    no_grad_context.__exit__.return_value = None
    fake_torch.no_grad = MagicMock(return_value=no_grad_context)

    fake_functional = SimpleNamespace(log_softmax=MagicMock())
    fake_nn = SimpleNamespace(functional=fake_functional)
    fake_torch.nn = fake_nn

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 99

    input_ids = MagicMock()
    input_ids.shape = (1, 3)
    input_ids.to.return_value = input_ids
    tokenizer.return_value = {"input_ids": input_ids}

    generated_ids = [[11, 12, 13, 21, 22]]
    tokenizer.decode.return_value = " generated text"

    prompt_target_ids = MagicMock()
    prompt_target_ids.shape = (1, 4)
    prompt_target_ids.to.return_value = prompt_target_ids

    def encode_side_effect(
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
    ) -> object:
        del add_special_tokens
        if return_tensors == "pt":
            return prompt_target_ids
        mapping = {
            "Prompt": [11, 12],
            " Target": [21, 22],
            "Prompt Target": [101, 11, 21, 22],
        }
        return mapping[text]

    tokenizer.encode.side_effect = encode_side_effect

    generated_logits = MagicMock()
    model_outputs = SimpleNamespace(logits=generated_logits)

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = generated_ids
    model.return_value = model_outputs
    model.eval.return_value = None

    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model

    tokenizer_cls = MagicMock()
    tokenizer_cls.from_pretrained.return_value = tokenizer

    score_a = MagicMock()
    score_a.item.return_value = -0.25
    score_b = MagicMock()
    score_b.item.return_value = -0.75

    log_probs = MagicMock()
    log_probs.shape = (4, 100)
    log_probs.__getitem__.side_effect = lambda key: {
        (1, 21): score_a,
        (2, 22): score_b,
    }[key]
    fake_functional.log_softmax.return_value = log_probs

    transformers_module = SimpleNamespace(
        AutoTokenizer=tokenizer_cls,
        AutoModelForCausalLM=model_cls,
    )

    with patch.dict(
        "sys.modules",
        {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "torch.nn.functional": fake_functional,
            "transformers": transformers_module,
        },
    ):
        hf_model = hf_model_cls(pretrained="mock/model", dtype="float16")
        generations = hf_model.generate(["Prompt"], max_tokens=2, temperature=0.0)
        scores = hf_model.loglikelihood(["Prompt"], [" Target"])

    tokenizer_cls.from_pretrained.assert_called_once_with("mock/model")
    model_cls.from_pretrained.assert_called_once_with("mock/model", torch_dtype="float16")
    model.eval.assert_called_once_with()
    model.generate.assert_called_once_with(
        input_ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=99,
        temperature=0.0,
    )
    fake_functional.log_softmax.assert_called_once_with(generated_logits[0], dim=-1)
    assert tokenizer.pad_token == "<eos>"
    assert hf_model.model_name == "mock/model"
    assert hf_model.supports_loglikelihood() is True
    assert generations == [" generated text"]
    assert scores == [-1.0]


@pytest.mark.slow
def test_hf_model_generate_with_tiny_gpt2() -> None:
    hf_model = cast(Callable[..., Model], ModelRegistry.get("hf"))
    model = hf_model(pretrained="sshleifer/tiny-gpt2")

    results = model.generate(["Halo dunia"], max_tokens=10)

    assert len(results) == 1
    assert isinstance(results[0], str)


@pytest.mark.slow
def test_hf_model_loglikelihood_with_tiny_gpt2() -> None:
    hf_model = cast(Callable[..., Model], ModelRegistry.get("hf"))
    model = hf_model(pretrained="sshleifer/tiny-gpt2")

    scores = model.loglikelihood(["Jakarta adalah"], ["ibu kota Indonesia"])

    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert scores[0] < 0


@pytest.mark.slow
def test_hf_model_model_name_property() -> None:
    hf_model = cast(Callable[..., Model], ModelRegistry.get("hf"))
    model = hf_model(pretrained="sshleifer/tiny-gpt2")

    assert model.model_name == "sshleifer/tiny-gpt2"


@pytest.mark.slow
def test_hf_model_supports_loglikelihood() -> None:
    hf_model = cast(Callable[..., Model], ModelRegistry.get("hf"))
    model = hf_model(pretrained="sshleifer/tiny-gpt2")

    assert model.supports_loglikelihood() is True
