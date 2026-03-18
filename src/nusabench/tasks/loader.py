# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from nusabench.tasks.base import TaskConfig


def load_task_config(path: str | Path) -> TaskConfig:
    path = Path(path)
    try:
        with path.open() as f:
            data = cast(object, yaml.safe_load(f))
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data)}")
    config_data = dict(data)
    required = [
        "task",
        "dataset_path",
        "output_type",
        "test_split",
        "doc_to_text",
        "doc_to_target",
        "metric_list",
    ]
    for field in required:
        if field not in config_data:
            raise ValueError(f"Missing required field '{field}' in {path}")
    return TaskConfig(
        task=cast(str, config_data["task"]),
        dataset_path=cast(str, config_data["dataset_path"]),
        dataset_name=cast(str | None, config_data.get("dataset_name")),
        output_type=cast(str, config_data["output_type"]),
        train_split=cast(str | None, config_data.get("train_split")),
        validation_split=cast(str | None, config_data.get("validation_split")),
        test_split=cast(str, config_data["test_split"]),
        doc_to_text=cast(str, config_data["doc_to_text"]),
        doc_to_target=cast(str, config_data["doc_to_target"]),
        metric_list=cast(list[dict[str, object]], config_data["metric_list"]),
        num_fewshot=cast(int, config_data.get("num_fewshot", 0)),
        description=cast(str, config_data.get("description", "")),
        metadata=cast(dict[str, object], config_data.get("metadata", {})),
        target_choices=cast(list[str] | None, config_data.get("target_choices")),
        generation_kwargs=cast(dict[str, object], config_data.get("generation_kwargs", {})),
        preprocess_fn=cast(str | None, config_data.get("preprocess_fn")),
    )
