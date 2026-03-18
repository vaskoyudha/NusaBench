from __future__ import annotations

import logging
from pathlib import Path

from nusabench.tasks.base import Task, TaskConfig
from nusabench.tasks.loader import load_task_config

logger = logging.getLogger(__name__)
_registry: dict[str, Task] = {}


class TaskRegistry:
    @staticmethod
    def get(name: str) -> Task:
        if name not in _registry:
            raise KeyError(f"Task '{name}' not found. Available: {list(_registry.keys())}")
        return _registry[name]

    @staticmethod
    def list() -> list[str]:
        return sorted(_registry.keys())

    @staticmethod
    def register(task: Task) -> None:
        _registry[task.config.task] = task


def _auto_discover_tasks() -> None:
    configs_dir = Path(__file__).parent / "configs"
    if not configs_dir.exists():
        return
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        try:
            config = load_task_config(yaml_path)
            _registry[config.task] = Task(config)
            logger.debug(f"Auto-registered task: {config.task}")
        except Exception as e:
            logger.warning(f"Skipping malformed task config {yaml_path.name}: {e}")

_auto_discover_tasks()

__all__ = ["Task", "TaskConfig", "TaskRegistry", "load_task_config"]
