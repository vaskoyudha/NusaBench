from __future__ import annotations

import json
from pathlib import Path

import pytest

from nusabench.reporting import export_json, export_markdown, print_results
from nusabench.results import EvaluationResult, TaskResult


@pytest.fixture()
def mock_result() -> EvaluationResult:
    return EvaluationResult(
        results={
            "test_task": TaskResult(
                task_name="test_task",
                metrics={"accuracy": 0.9},
                num_samples=10,
                model_name="dummy",
            ),
        },
        model="dummy",
    )


def test_json_export_creates_file(mock_result: EvaluationResult, tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    export_json(mock_result, out)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "model" in data
    assert "results" in data
    assert "nusabench_version" in data
    assert "timestamp" in data


def test_json_export_metric_value(mock_result: EvaluationResult, tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    export_json(mock_result, out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["results"]["test_task"]["accuracy"] == pytest.approx(0.9)


def test_json_export_ensure_ascii_false(tmp_path: Path) -> None:
    indo_result = EvaluationResult(
        results={
            "sentimen_bahasa": TaskResult(
                task_name="sentimen_bahasa",
                metrics={"akurasi": 0.75},
                num_samples=5,
                model_name="dummy",
            ),
        },
        model="dummy",
    )
    out = tmp_path / "indo.json"
    export_json(indo_result, out)

    raw = out.read_text(encoding="utf-8")
    assert "sentimen_bahasa" in raw
    assert "akurasi" in raw


def test_markdown_export_creates_table(mock_result: EvaluationResult, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    export_markdown(mock_result, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "| Task" in content
    assert "| Metric" in content
    assert "| Score" in content


def test_markdown_export_contains_data(mock_result: EvaluationResult, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    export_markdown(mock_result, out)

    content = out.read_text(encoding="utf-8")
    assert "test_task" in content
    assert "accuracy" in content
    assert "0.9000" in content


def test_console_print_no_error(mock_result: EvaluationResult) -> None:
    print_results(mock_result)


def test_reporting_module_imports() -> None:
    from nusabench.reporting import export_json, export_markdown, print_results  # noqa: F811

    assert callable(export_json)
    assert callable(export_markdown)
    assert callable(print_results)
