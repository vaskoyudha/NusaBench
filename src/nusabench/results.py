from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskResult:
    task_name: str
    metrics: dict[str, float]
    num_samples: int
    model_name: str


@dataclass
class EvaluationResult:
    results: dict[str, TaskResult]
    model: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "results": {
                task_name: {
                    "task_name": task_result.task_name,
                    "metrics": task_result.metrics,
                    "num_samples": task_result.num_samples,
                    "model_name": task_result.model_name,
                }
                for task_name, task_result in self.results.items()
            },
            "metadata": self.metadata,
        }
