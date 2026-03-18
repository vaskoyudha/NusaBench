from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"


def evaluate(
    model: str,
    tasks: list[str],
    model_args: str | dict[str, str] | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> EvaluationResult:
    from nusabench.evaluator import Evaluator
    from nusabench.models.base import Model  # noqa: PLC0415
    from nusabench.tasks import TaskRegistry  # noqa: PLC0415
    from nusabench.tasks.base import Task  # noqa: PLC0415
    from nusabench.utils.config import NusaBenchConfig  # noqa: PLC0415

    model_registry = import_module("nusabench.models").ModelRegistry

    parsed_args: dict[str, object] = {}
    if isinstance(model_args, str) and model_args:
        for pair in model_args.split(","):
            key, _, value = pair.partition("=")
            parsed_args[key.strip()] = value.strip()
    elif isinstance(model_args, dict):
        parsed_args = dict(model_args)

    try:
        model_cls = model_registry.get(model)
    except KeyError as exc:
        available = model_registry.list()
        raise ValueError(f"Unknown model '{model}'. Available: {available}") from exc
    model_instance: Model = model_cls(**parsed_args)

    task_instances: list[Task] = []
    for task_name in tasks:
        try:
            task_instances.append(TaskRegistry.get(task_name))
        except KeyError as exc:
            available = TaskRegistry.list()
            raise ValueError(f"Unknown task '{task_name}'. Available: {available}") from exc

    config = NusaBenchConfig(verbose=verbose, limit=limit)
    evaluator = Evaluator(model=model_instance, tasks=task_instances, config=config)
    return evaluator.evaluate()


def list_tasks() -> list[str]:
    from nusabench.tasks import TaskRegistry  # noqa: PLC0415

    return TaskRegistry.list()


from nusabench.results import EvaluationResult, TaskResult  # noqa: E402, F401

__all__ = [
    "__version__",
    "evaluate",
    "list_tasks",
    "EvaluationResult",
    "TaskResult",
]
