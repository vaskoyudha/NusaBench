from __future__ import annotations

from pathlib import Path

from nusabench.tasks import Task, TaskRegistry, load_task_config


def test_toxicity_yaml_loads() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/toxicity_id.yaml"
    config = load_task_config(yaml_path)

    assert config.task == "toxicity_id"
    assert config.output_type == "multiple_choice"
    assert config.target_choices is not None
    assert len(config.target_choices) == 2


def test_toxicity_metrics_and_template() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/toxicity_id.yaml"
    config = load_task_config(yaml_path)
    assert any(m["metric"] == "accuracy" for m in config.metric_list)
    assert any(m["metric"] == "f1" for m in config.metric_list)

    task = Task(config)
    doc = {"text": "Kamu jelek"}
    prompt = task.format_prompt(doc)
    assert "Teks: Kamu jelek" in prompt
    assert "mengandung bahasa kasar" in prompt


def test_task_registry_register_and_get() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/toxicity_id.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    TaskRegistry.register(task)
    got = TaskRegistry.get("toxicity_id")
    assert got is not None
    assert got.config.task == "toxicity_id"
