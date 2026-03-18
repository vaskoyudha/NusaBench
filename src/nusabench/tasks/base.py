# pyright: reportExplicitAny=false, reportAny=false
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import jinja2

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    task: str
    dataset_path: str
    dataset_name: str | None
    output_type: str
    train_split: str | None
    validation_split: str | None
    test_split: str
    doc_to_text: str
    doc_to_target: str
    metric_list: list[dict[str, Any]]
    num_fewshot: int = 0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    target_choices: list[str] | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    preprocess_fn: str | None = None


class Task:
    def __init__(self, config: TaskConfig) -> None:
        self.config: TaskConfig = config
        self._jinja_env: jinja2.Environment = jinja2.Environment()

    def load_dataset(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        module = import_module("nusabench.utils.data")
        load_hf_dataset = cast(
            Callable[..., list[dict[str, Any]]],
            module.load_hf_dataset,
        )

        return load_hf_dataset(
            path=self.config.dataset_path,
            name=self.config.dataset_name,
            split=split,
            limit=limit,
        )

    def format_prompt(self, doc: dict[str, Any]) -> str:
        tmpl = self._jinja_env.from_string(self.config.doc_to_text)
        return tmpl.render(**doc)

    def format_target(self, doc: dict[str, Any]) -> str:
        tmpl = self._jinja_env.from_string(self.config.doc_to_target)
        return tmpl.render(**doc)

    def preprocess_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        if self.config.preprocess_fn is None:
            return doc
        module_path, func_name = self.config.preprocess_fn.rsplit(".", 1)
        module = import_module(module_path)
        fn = cast(Callable[[dict[str, Any]], dict[str, Any]], getattr(module, func_name))
        return fn(doc)

    def get_choices(self) -> list[str] | None:
        return self.config.target_choices
