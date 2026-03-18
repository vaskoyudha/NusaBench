"""Google Gemini API model backend for NusaBench."""

from __future__ import annotations

import logging
import os
import time
from importlib import import_module
from typing import Any

from typing_extensions import override

from nusabench.models.base import Model, register_model

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAYS = [1.0, 2.0, 4.0]


@register_model("gemini")
class GeminiModel(Model):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY environment variable not set and no api_key provided")

        try:
            self._genai = import_module("google.generativeai")
        except ImportError as exc:
            raise ImportError(
                "Gemini backend requires [gemini] extras. Install with: uv sync --extra gemini"
            ) from exc

        self._genai.configure(api_key=resolved_key)
        self._model = model

    @property
    @override
    def model_name(self) -> str:
        return f"gemini:{self._model}"

    @override
    def supports_loglikelihood(self) -> bool:
        return False

    @override
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> list[str]:
        del max_tokens, kwargs

        results: list[str] = []
        for prompt in prompts:
            text = self._generate_single(prompt)
            results.append(text)
        return results

    def _generate_single(self, prompt: str) -> str:
        try:
            api_core_exceptions = import_module("google.api_core.exceptions")
            resource_exhausted = api_core_exceptions.ResourceExhausted
        except ImportError:
            resource_exhausted = None

        last_exception: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                generative_model = self._genai.GenerativeModel(self._model)
                response = generative_model.generate_content(prompt)
                return str(response.text)
            except Exception as exc:
                if resource_exhausted is not None and isinstance(exc, resource_exhausted):
                    last_exception = exc
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_DELAYS[attempt]
                        logger.warning(
                            "Rate limited (attempt %d/%d), retrying in %.1fs...",
                            attempt + 1,
                            _MAX_RETRIES,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                raise RuntimeError(f"Gemini API error: {exc}") from exc

        raise RuntimeError(f"Gemini API error: {last_exception}") from last_exception
