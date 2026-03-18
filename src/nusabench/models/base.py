from __future__ import annotations

from abc import ABC, abstractmethod

_registry: dict[str, type[Model]] = {}


def register_model(name: str):
    def decorator(cls: type[Model]) -> type[Model]:
        _registry[name] = cls
        return cls

    return decorator


class Model(ABC):
    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        **kwargs: object,
    ) -> list[str]:
        ...

    def loglikelihood(self, prompts: list[str], targets: list[str]) -> list[float]:
        del prompts, targets
        raise NotImplementedError("This model does not support loglikelihood scoring")

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    def supports_loglikelihood(self) -> bool:
        return False


class ModelRegistry:
    @staticmethod
    def get(name: str) -> type[Model]:
        if name not in _registry:
            raise KeyError(f"Model '{name}' not found. Available: {list(_registry.keys())}")
        return _registry[name]

    @staticmethod
    def list() -> list[str]:
        return sorted(_registry.keys())
