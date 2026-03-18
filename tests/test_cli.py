from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nusabench.cli import app
from nusabench.tasks import Task, TaskRegistry
from nusabench.tasks.loader import load_task_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"

SAMPLE_DATASET: list[dict[str, object]] = [
    {"text": "contoh 1", "label": "positif"},
    {"text": "contoh 2", "label": "positif"},
]

runner = CliRunner()


@pytest.fixture(autouse=True)
def _register_dummy_task() -> None:
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))
    TaskRegistry.register(task)


def test_help_output() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output
    assert "list-tasks" in result.output
    assert "list-models" in result.output


def test_list_models_shows_registered_backends() -> None:
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "dummy" in result.output


def test_list_tasks_shows_registered_tasks() -> None:
    result = runner.invoke(app, ["list-tasks"])
    assert result.exit_code == 0
    assert "dummy_task" in result.output


def test_evaluate_dummy_model_creates_output(tmp_path: Path) -> None:
    output_file = str(tmp_path / "results.json")
    with patch("nusabench.tasks.base.Task.load_dataset", return_value=SAMPLE_DATASET):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy",
                "--task",
                "dummy_task",
                "--limit",
                "2",
                "--output",
                output_file,
            ],
        )
    assert result.exit_code == 0, result.output
    import json

    with open(output_file, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "results" in data
    assert "dummy_task" in data["results"]


def test_evaluate_invalid_model_shows_error() -> None:
    with patch("nusabench.tasks.base.Task.load_dataset", return_value=SAMPLE_DATASET):
        result = runner.invoke(
            app,
            ["evaluate", "--model", "nonexistent_model", "--task", "dummy_task", "--limit", "1"],
        )
    assert result.exit_code == 1
    assert "nonexistent_model" in result.output


def test_evaluate_invalid_task_shows_error() -> None:
    with patch("nusabench.tasks.base.Task.load_dataset", return_value=SAMPLE_DATASET):
        result = runner.invoke(
            app,
            ["evaluate", "--model", "dummy", "--task", "nonexistent_task", "--limit", "1"],
        )
    assert result.exit_code == 1
    assert "nonexistent_task" in result.output
