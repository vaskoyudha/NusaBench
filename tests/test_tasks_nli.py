from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nusabench.tasks import Task
from nusabench.tasks.loader import load_task_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIGS_DIR = Path(__file__).parent.parent / "src" / "nusabench" / "tasks" / "configs"


class TestNliWreteConfig:
    """Tests for NLI WReTe task configuration and loading."""

    def test_nli_wrete_yaml_exists(self) -> None:
        """Verify the YAML config file exists."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        assert yaml_path.exists(), f"nli_wrete.yaml not found at {yaml_path}"

    def test_nli_wrete_loads_successfully(self) -> None:
        """Verify the YAML config can be loaded."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config is not None

    def test_nli_wrete_config_task_name(self) -> None:
        """Verify task name is nli_wrete."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.task == "nli_wrete"

    def test_nli_wrete_config_output_type(self) -> None:
        """Verify output_type is multiple_choice."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.output_type == "multiple_choice"

    def test_nli_wrete_config_dataset_path(self) -> None:
        """Verify dataset_path is indonlp/indonlu."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.dataset_path == "indonlp/indonlu"

    def test_nli_wrete_config_dataset_name(self) -> None:
        """Verify dataset_name is wrete."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.dataset_name == "wrete"

    def test_nli_wrete_config_splits(self) -> None:
        """Verify splits are correctly configured."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.test_split == "test"
        assert config.validation_split == "validation"

    def test_nli_wrete_config_target_choices(self) -> None:
        """Verify target_choices has exactly 2 items."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.target_choices == ["entailment", "not_entailment"]
        assert len(config.target_choices) == 2

    def test_nli_wrete_config_metric_list(self) -> None:
        """Verify metric_list has 2 items."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert len(config.metric_list) == 2
        assert config.metric_list[0]["metric"] == "accuracy"
        assert config.metric_list[0]["aggregation"] == "mean"
        assert config.metric_list[1]["metric"] == "f1"
        assert config.metric_list[1]["aggregation"] == "binary"

    def test_nli_wrete_config_doc_to_text_template(self) -> None:
        """Verify doc_to_text template."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        expected = (
            "Premis: {{premise}}\nHipotesis: {{hypothesis}}\nRelasi antara premis dan hipotesis:"
        )
        assert config.doc_to_text == expected

    def test_nli_wrete_config_doc_to_target_template(self) -> None:
        """Verify doc_to_target template."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.doc_to_target == "{{label}}"

    def test_nli_wrete_config_metadata(self) -> None:
        """Verify metadata is correctly set."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        assert config.metadata["source"] == "IndoNLU"
        assert config.metadata["language"] == "id"
        assert config.metadata["task_type"] == "classification"
        assert config.metadata["num_classes"] == 2

    def test_nli_wrete_config_description(self) -> None:
        """Verify description is set."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        expected = "Textual entailment on Indonesian text (WReTe from IndoNLU)"
        assert config.description == expected


class TestNliWreteTask:
    """Tests for NLI WReTe task runtime behavior."""

    def test_nli_wrete_task_creation(self) -> None:
        """Verify a Task can be created from the config."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        task = Task(config)
        assert task is not None
        assert task.config.task == "nli_wrete"

    def test_nli_wrete_format_prompt_with_sample_doc(self) -> None:
        """Verify doc_to_text template renders correctly."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        task = Task(config)
        sample_doc = {
            "premise": "Ini adalah premis",
            "hypothesis": "Ini hipotesis",
            "label": "entailment",
        }
        prompt = task.format_prompt(sample_doc)
        assert "Ini adalah premis" in prompt
        assert "Ini hipotesis" in prompt
        assert "Premis:" in prompt
        assert "Hipotesis:" in prompt

    def test_nli_wrete_format_target_with_sample_doc(self) -> None:
        """Verify doc_to_target template renders correctly."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        task = Task(config)
        sample_doc = {"label": "entailment"}
        target = task.format_target(sample_doc)
        assert target == "entailment"

    def test_nli_wrete_get_choices(self) -> None:
        """Verify get_choices returns the target_choices."""
        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        task = Task(config)
        choices = task.get_choices()
        assert choices == ["entailment", "not_entailment"]

    def test_nli_wrete_load_dataset_with_mocking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify load_dataset works with mocked utils module."""

        def fake_load_hf_dataset(
            *,
            path: str,
            name: str | None,
            split: str,
            limit: int | None,
        ) -> list[dict[str, object]]:
            assert path == "indonlp/indonlu"
            assert name == "wrete"
            assert split == "test"
            return [
                {
                    "premise": "Semua kucing adalah hewan",
                    "hypothesis": "Kucing adalah hewan",
                    "label": "entailment",
                }
            ]

        utils_module = ModuleType("nusabench.utils")
        data_module = ModuleType("nusabench.utils.data")
        data_module.load_hf_dataset = fake_load_hf_dataset

        monkeypatch.setitem(sys.modules, "nusabench.utils", utils_module)
        monkeypatch.setitem(sys.modules, "nusabench.utils.data", data_module)
        importlib.invalidate_caches()

        yaml_path = CONFIGS_DIR / "nli_wrete.yaml"
        config = load_task_config(yaml_path)
        task = Task(config)
        dataset = task.load_dataset("test")

        assert len(dataset) == 1
        assert dataset[0]["premise"] == "Semua kucing adalah hewan"
        assert dataset[0]["label"] == "entailment"
