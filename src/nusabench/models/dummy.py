from __future__ import annotations

from typing_extensions import override

from nusabench.models.base import Model, register_model


@register_model("dummy")
class DummyModel(Model):
    @override
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        **kwargs: object,
    ) -> list[str]:
        del max_tokens, kwargs
        return ["dummy response"] * len(prompts)

    @override
    def loglikelihood(self, prompts: list[str], targets: list[str]) -> list[float]:
        del targets
        return [-1.0] * len(prompts)

    @property
    @override
    def model_name(self) -> str:
        return "dummy"

    @override
    def supports_loglikelihood(self) -> bool:
        return True
