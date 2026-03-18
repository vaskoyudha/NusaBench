from __future__ import annotations

from pathlib import Path

from nusabench.tasks import Task, TaskRegistry
from nusabench.tasks.loader import load_task_config

CONFIGS_DIR = Path(__file__).parent.parent / "src" / "nusabench" / "tasks" / "configs"


def test_summarization_config_loads() -> None:
    config = load_task_config(CONFIGS_DIR / "summarization_indosum.yaml")

    assert config.task == "summarization_indosum"


def test_summarization_output_type_and_generation_kwargs() -> None:
    config = load_task_config(CONFIGS_DIR / "summarization_indosum.yaml")

    assert config.output_type == "generate_until"
    assert config.generation_kwargs["max_tokens"] == 512


def test_summarization_metrics_and_choices() -> None:
    config = load_task_config(CONFIGS_DIR / "summarization_indosum.yaml")

    metrics = {m["metric"] for m in config.metric_list}
    assert "rouge" in metrics
    assert config.target_choices is None


def test_summarization_doc_template_and_registry() -> None:
    config = load_task_config(CONFIGS_DIR / "summarization_indosum.yaml")
    task_instance = Task(config)

    # ensure doc_to_text uses expected field (text for XL-Sum)
    sample_doc = {"text": "Ini adalah artikel berita singkat."}
    prompt = task_instance.format_prompt(sample_doc)
    assert "Rangkum teks berikut" in prompt
    assert "Ini adalah artikel berita singkat." in prompt

    # register and retrieve
    TaskRegistry.register(task_instance)
    task = TaskRegistry.get("summarization_indosum")
    assert isinstance(task, Task)
    assert task.config.task == "summarization_indosum"
