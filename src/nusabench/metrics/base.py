from __future__ import annotations

from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def compute(self, predictions: list[object], references: list[object]) -> dict[str, float]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
