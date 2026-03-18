from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from nusabench import tasks as tasks_module
from nusabench.tasks import Task, TaskConfig, TaskRegistry
from nusabench.tasks.loader import load_task_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def reload_tasks_with_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    config_dir: Path,
) -> ModuleType:
    original_exists = Path.exists
    original_glob = Path.glob

    def patched_exists(path: Path) -> bool:
        if path == Path(tasks_module.__file__).parent / "configs":
            return True
        return original_exists(path)

    def patched_glob(path: Path, pattern: str):
        if path == Path(tasks_module.__file__).parent / "configs":
            return config_dir.glob(pattern)
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "exists", patched_exists)
    monkeypatch.setattr(Path, "glob", patched_glob)
    return importlib.reload(tasks_module)


def test_load_task_config_parses_expected_fields() -> None:
    config = load_task_config(FIXTURES_DIR / "dummy_task.yaml")

    assert config.task == "dummy_task"
    assert config.dataset_path == "dummy/dataset"
    assert config.dataset_name is None
    assert config.output_type == "generate_until"
    assert config.train_split == "train"
    assert config.validation_split == "validation"
    assert config.test_split == "test"
    assert config.doc_to_text == "Teks: {{text}}"
    assert config.doc_to_target == "{{label}}"
    assert config.metric_list == [{"metric": "exact_match", "aggregation": "mean"}]
    assert config.num_fewshot == 0
    assert config.description == "A dummy task for testing"
    assert config.target_choices == ["positif", "negatif", "netral"]
    assert config.generation_kwargs == {"max_tokens": 128}
    assert config.preprocess_fn is None


def test_load_task_config_defaults_preprocess_fn_when_absent(tmp_path: Path) -> None:
    config_path = tmp_path / "task.yaml"
    _ = config_path.write_text(
        "\n".join(
            [
                "task: no_preprocess",
                "dataset_path: dummy/dataset",
                "output_type: generate_until",
                "test_split: test",
                'doc_to_text: "{{text}}"',
                'doc_to_target: "{{label}}"',
                "metric_list:",
                "  - metric: exact_match",
                "    aggregation: mean",
            ]
        ),
        encoding="utf-8",
    )

    config = load_task_config(config_path)

    assert config.preprocess_fn is None


def test_task_formats_prompt_and_target_with_jinja() -> None:
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))
    doc = {"text": "contoh", "label": "positif"}

    assert task.format_prompt(doc) == "Teks: contoh"
    assert task.format_target(doc) == "positif"


def test_preprocess_doc_returns_original_doc_without_preprocess_fn() -> None:
    doc = {"text": "contoh", "label": "netral"}
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))

    assert task.preprocess_doc(doc) is doc


def test_get_choices_returns_target_choices() -> None:
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))

    assert task.get_choices() == ["positif", "negatif", "netral"]


def test_invalid_yaml_raises_value_error(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.yaml"
    _ = invalid_path.write_text("task: [broken", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid YAML in .*invalid\.yaml"):
        _ = load_task_config(invalid_path)


def test_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    _ = missing_path.write_text(
        "\n".join(
            [
                "task: missing_metric",
                "dataset_path: dummy/dataset",
                "output_type: generate_until",
                "test_split: test",
                'doc_to_text: "{{text}}"',
                'doc_to_target: "{{label}}"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required field 'metric_list'"):
        _ = load_task_config(missing_path)


def test_task_registry_lists_registered_tasks() -> None:
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))
    TaskRegistry.register(task)

    assert "dummy_task" in TaskRegistry.list()


def test_task_registry_get_raises_for_unknown_task() -> None:
    with pytest.raises(KeyError, match="Task 'nonexistent' not found"):
        _ = TaskRegistry.get("nonexistent")


def test_task_registry_register_supports_manual_registration_from_loaded_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "temp_task.yaml"
    _ = config_path.write_text(
        "\n".join(
            [
                "task: temp_task",
                "dataset_path: dummy/dataset",
                "output_type: multiple_choice",
                "test_split: test",
                'doc_to_text: "Prompt: {{text}}"',
                'doc_to_target: "{{label}}"',
                "metric_list:",
                "  - metric: exact_match",
                "    aggregation: mean",
            ]
        ),
        encoding="utf-8",
    )
    config = load_task_config(config_path)

    TaskRegistry.register(Task(config))

    assert "temp_task" in TaskRegistry.list()
    assert TaskRegistry.get("temp_task").config.task == "temp_task"


def test_malformed_yaml_error_message_is_clear(tmp_path: Path) -> None:
    malformed_path = tmp_path / "malformed.yaml"
    _ = malformed_path.write_text("metric_list: [", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        _ = load_task_config(malformed_path)

    assert "Invalid YAML in" in str(exc_info.value)
    assert "malformed.yaml" in str(exc_info.value)


def test_auto_discover_tasks_registers_valid_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _ = (config_dir / "auto.yaml").write_text(
        "\n".join(
            [
                "task: auto_task",
                "dataset_path: dummy/dataset",
                "output_type: generate_until",
                "test_split: test",
                'doc_to_text: "Auto {{text}}"',
                'doc_to_target: "{{label}}"',
                "metric_list:",
                "  - metric: exact_match",
                "    aggregation: mean",
            ]
        ),
        encoding="utf-8",
    )

    reloaded_tasks = reload_tasks_with_config_dir(monkeypatch, config_dir)

    reloaded_registry = cast(type[TaskRegistry], reloaded_tasks.TaskRegistry)

    assert reloaded_registry.list() == ["auto_task"]


def test_auto_discover_tasks_logs_warning_for_malformed_yaml(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _ = (config_dir / "broken.yaml").write_text("task: [broken", encoding="utf-8")

    with caplog.at_level("WARNING"):
        reloaded_tasks = reload_tasks_with_config_dir(monkeypatch, config_dir)

    reloaded_registry = cast(type[TaskRegistry], reloaded_tasks.TaskRegistry)

    assert reloaded_registry.list() == []
    assert "Skipping malformed task config broken.yaml" in caplog.text


def test_load_dataset_uses_mocked_utils_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_hf_dataset(
        *,
        path: str,
        name: str | None,
        split: str,
        limit: int | None,
    ) -> list[dict[str, object]]:
        assert path == "dummy/dataset"
        assert name is None
        assert split == "test"
        assert limit == 2
        return [{"text": "contoh", "label": "positif"}]

    utils_module = ModuleType("nusabench.utils")
    data_module = ModuleType("nusabench.utils.data")
    data_module.load_hf_dataset = fake_load_hf_dataset

    monkeypatch.setitem(sys.modules, "nusabench.utils", utils_module)
    monkeypatch.setitem(sys.modules, "nusabench.utils.data", data_module)
    importlib.invalidate_caches()

    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))

    assert task.load_dataset("test", limit=2) == [{"text": "contoh", "label": "positif"}]


def test_task_config_supports_explicit_construction() -> None:
    config = TaskConfig(
        task="manual",
        dataset_path="dummy/dataset",
        dataset_name=None,
        output_type="generate_until",
        train_split=None,
        validation_split=None,
        test_split="test",
        doc_to_text="{{text}}",
        doc_to_target="{{label}}",
        metric_list=[{"metric": "exact_match", "aggregation": "mean"}],
    )

    assert Task(config).format_prompt({"text": "manual", "label": "ok"}) == "manual"
