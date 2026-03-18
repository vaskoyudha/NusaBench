"""HuggingFace Transformers model backend for NusaBench."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

from typing_extensions import override

from nusabench.models.base import Model, register_model

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedFunction=false

logger = logging.getLogger(__name__)


def _parse_model_args(model_args: str) -> dict[str, str]:
    if not model_args:
        return {}

    result: dict[str, str] = {}
    for pair in model_args.split(","):
        key, _, value = pair.partition("=")
        result[key.strip()] = value.strip()
    return result


@register_model("hf")
class HuggingFaceModel(Model):
    def __init__(
        self,
        pretrained: str,
        device: str = "cpu",
        dtype: str = "auto",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        model_args = _parse_model_args(str(kwargs.pop("model_args", "")))
        del kwargs

        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
            auto_model_for_causal_lm = transformers.AutoModelForCausalLM
            auto_tokenizer = transformers.AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace backend requires [hf] extras. Install with: uv sync --extra hf"
            ) from exc

        self._pretrained = pretrained
        self._device = device
        self._batch_size = batch_size

        logger.info("Loading model: %s", pretrained)

        self._tokenizer = auto_tokenizer.from_pretrained(pretrained)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        torch_dtype: Any = torch.float32
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "auto":
            torch_dtype = "auto"

        self._model = auto_model_for_causal_lm.from_pretrained(
            pretrained,
            torch_dtype=torch_dtype,
            **model_args,
        )
        self._model.eval()

        if device not in {"cpu", "auto"}:
            self._model = self._model.to(device)

        self._torch = torch

    @property
    @override
    def model_name(self) -> str:
        return self._pretrained

    @override
    def supports_loglikelihood(self) -> bool:
        return True

    @override
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> list[str]:
        results: list[str] = []
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            if hasattr(self._model, "device") and str(self._model.device) != "cpu":
                input_ids = input_ids.to(self._model.device)

            with self._torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **{key: value for key, value in kwargs.items() if key != "max_tokens"},
                )

            new_tokens = output_ids[0][input_ids.shape[-1] :]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text)

        return results

    @override
    def loglikelihood(self, prompts: list[str], targets: list[str]) -> list[float]:
        functional = import_module("torch.nn.functional")

        scores: list[float] = []
        for prompt, target in zip(prompts, targets, strict=False):
            self._tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self._tokenizer.encode(target, add_special_tokens=False)

            if not target_ids:
                scores.append(0.0)
                continue

            full_ids = self._tokenizer.encode(
                prompt + target,
                return_tensors="pt",
                add_special_tokens=True,
            )
            if hasattr(self._model, "device") and str(self._model.device) != "cpu":
                full_ids = full_ids.to(self._model.device)

            with self._torch.no_grad():
                outputs = self._model(full_ids)
                logits = outputs.logits

            log_probs = functional.log_softmax(logits[0], dim=-1)

            full_len = full_ids.shape[1]
            target_start = full_len - len(target_ids)
            if target_start <= 0:
                target_start = 1

            score = 0.0
            for index, token_id in enumerate(target_ids):
                position = target_start - 1 + index
                if position < log_probs.shape[0]:
                    score += log_probs[position, token_id].item()

            scores.append(score)

        return scores
