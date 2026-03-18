from __future__ import annotations

from pathlib import Path

from nusabench.tasks import Task
from nusabench.tasks.loader import load_task_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_sentiment_smsa_yaml_loads() -> None:
    """Test that sentiment_smsa.yaml loads correctly."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)

    assert config.task == "sentiment_smsa"
    assert config.dataset_path == "indonlp/indonlu"
    assert config.dataset_name == "smsa"
    assert config.test_split == "test"
    assert config.validation_split == "validation"
    assert config.output_type == "multiple_choice"


def test_sentiment_smsa_doc_to_text_template() -> None:
    """Test doc_to_text template renders correctly."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    doc = {"text": "Makanan ini enak"}
    prompt = task.format_prompt(doc)

    assert "Teks: Makanan ini enak" in prompt
    assert "Sentimen dari teks di atas adalah:" in prompt


def test_sentiment_smsa_doc_to_target_template() -> None:
    """Test doc_to_target template renders correctly."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    doc = {"label": "positif"}
    target = task.format_target(doc)

    assert target == "positif"


def test_sentiment_smsa_target_choices() -> None:
    """Test target_choices are exactly 3 sentiment labels."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)

    assert config.target_choices is not None
    assert len(config.target_choices) == 3
    assert config.target_choices == ["positif", "negatif", "netral"]


def test_sentiment_smsa_metrics() -> None:
    """Test metric_list has accuracy and f1."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)

    assert len(config.metric_list) == 2
    metrics = {m["metric"]: m["aggregation"] for m in config.metric_list}
    assert metrics["accuracy"] == "mean"
    assert metrics["f1"] == "macro"


def test_sentiment_smsa_metadata() -> None:
    """Test metadata contains expected fields."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)

    assert config.metadata["source"] == "IndoNLU"
    assert config.metadata["language"] == "id"
    assert config.metadata["task_type"] == "classification"
    assert config.metadata["num_classes"] == 3


def test_sentiment_smsa_description() -> None:
    """Test description is set."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)

    assert config.description != ""
    assert "Sentiment analysis" in config.description
    assert "Indonesian" in config.description


def test_sentiment_smsa_task_instantiation() -> None:
    """Test Task can be instantiated from sentiment_smsa config."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    assert task is not None
    assert isinstance(task, Task)
    assert task.config.task == "sentiment_smsa"
    assert task.config.dataset_path == "indonlp/indonlu"
    assert task.config.dataset_name == "smsa"
    assert task.config.output_type == "multiple_choice"


def test_sentiment_smsa_get_choices() -> None:
    """Test get_choices() returns sentiment labels."""
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/sentiment_smsa.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    choices = task.get_choices()

    assert choices is not None
    assert choices == ["positif", "negatif", "netral"]
