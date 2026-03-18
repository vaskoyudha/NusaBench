from __future__ import annotations

from pathlib import Path

from nusabench.tasks import Task, TaskRegistry
from nusabench.tasks.loader import load_task_config

CONFIGS_DIR = Path(__file__).parent.parent / "src" / "nusabench" / "tasks" / "configs"


def test_qa_facqa_config_loads() -> None:
    """Test that qa_facqa config loads from YAML file."""
    config_path = CONFIGS_DIR / "qa_facqa.yaml"
    config = load_task_config(config_path)

    assert config.task == "qa_facqa"


def test_qa_facqa_output_type() -> None:
    """Test that qa_facqa has generate_until output type."""
    config = load_task_config(CONFIGS_DIR / "qa_facqa.yaml")

    assert config.output_type == "generate_until"


def test_qa_facqa_dataset_config() -> None:
    """Test that qa_facqa dataset configuration is correct."""
    config = load_task_config(CONFIGS_DIR / "qa_facqa.yaml")

    assert config.dataset_path == "indonlp/indonlu"
    assert config.dataset_name == "facqa"


def test_qa_facqa_is_not_multiple_choice() -> None:
    """Test that qa_facqa does not have target_choices (not multiple choice)."""
    config = load_task_config(CONFIGS_DIR / "qa_facqa.yaml")

    assert config.target_choices is None


def test_qa_facqa_generation_kwargs() -> None:
    """Test that qa_facqa has proper generation_kwargs."""
    config = load_task_config(CONFIGS_DIR / "qa_facqa.yaml")

    assert config.generation_kwargs["max_tokens"] == 128
    assert config.generation_kwargs["stop_sequences"] == ["\n"]


def test_qa_facqa_metrics() -> None:
    """Test that qa_facqa has exact_match and f1 metrics."""
    config = load_task_config(CONFIGS_DIR / "qa_facqa.yaml")

    metrics = {m["metric"] for m in config.metric_list}
    assert "exact_match" in metrics
    assert "f1" in metrics


def test_qa_facqa_doc_to_text_template() -> None:
    """Test that qa_facqa doc_to_text template renders correctly."""
    task = Task(load_task_config(CONFIGS_DIR / "qa_facqa.yaml"))
    sample_doc = {"question": "Apa ibu kota Indonesia?"}

    prompt = task.format_prompt(sample_doc)

    assert "Pertanyaan: Apa ibu kota Indonesia?" in prompt
    assert "Jawaban:" in prompt


def test_qa_facqa_in_registry() -> None:
    """Test that qa_facqa task can be retrieved from TaskRegistry."""
    task = TaskRegistry.get("qa_facqa")

    assert isinstance(task, Task)
    assert task.config.task == "qa_facqa"
