from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

import nusabench as nb
from nusabench import EvaluationResult
from nusabench.cli import app
from nusabench.tasks.loader import load_task_config

CONFIGS_DIR = Path(nb.__file__).resolve().parent / "tasks" / "configs"

ALL_TASK_NAMES = [
    "sentiment_smsa",
    "nli_wrete",
    "qa_facqa",
    "ner_nergrit",
    "summarization_indosum",
    "mt_nusax",
    "toxicity_id",
    "cultural_indommu",
]

TASK_SAMPLE_DATASETS: dict[str, list[dict[str, object]]] = {
    "sentiment_smsa": [
        {"text": "Bagus sekali!", "label": 0},
        {"text": "Buruk sekali!", "label": 1},
        {"text": "Biasa saja.", "label": 2},
    ],
    "nli_wrete": [
        {"premise": "Kucing tidur.", "hypothesis": "Hewan tidur.", "label": 0},
        {"premise": "Langit biru.", "hypothesis": "Langit merah.", "label": 1},
        {"premise": "Air mengalir.", "hypothesis": "Sungai ada.", "label": 0},
    ],
    "qa_facqa": [
        {"question": "Siapa presiden pertama?", "answer": "Soekarno"},
        {"question": "Apa ibu kota Indonesia?", "answer": "Jakarta"},
        {"question": "Berapa provinsi?", "answer": "38"},
    ],
    "ner_nergrit": [
        {
            "tokens_joined": "Jokowi tinggal di Jakarta",
            "entities_formatted": "PER: Jokowi, LOC: Jakarta",
        },
        {
            "tokens_joined": "Google dibuat oleh Larry Page",
            "entities_formatted": "ORG: Google, PER: Larry Page",
        },
        {"tokens_joined": "Saya tinggal di Bandung", "entities_formatted": "LOC: Bandung"},
    ],
    "summarization_indosum": [
        {"text": "Indonesia memiliki banyak pulau.", "summary": "Indonesia berpulau banyak."},
        {"text": "Ekonomi tumbuh pesat tahun ini.", "summary": "Ekonomi tumbuh."},
        {"text": "Cuaca cerah hari ini di Jakarta.", "summary": "Cuaca cerah Jakarta."},
    ],
    "mt_nusax": [
        {"ind": "Selamat pagi.", "eng": "Good morning."},
        {"ind": "Terima kasih.", "eng": "Thank you."},
        {"ind": "Apa kabar?", "eng": "How are you?"},
    ],
    "toxicity_id": [
        {"text": "Kamu hebat!", "label": 0},
        {"text": "Kamu bodoh!", "label": 1},
        {"text": "Hari yang indah.", "label": 0},
    ],
    "cultural_indommu": [
        {
            "question": "Apa nama ibukota Indonesia?",
            "choice_a": "Jakarta",
            "choice_b": "Bandung",
            "choice_c": "Surabaya",
            "choice_d": "Medan",
            "answer": "A",
        },
        {
            "question": "Siapa proklamator Indonesia?",
            "choice_a": "Soeharto",
            "choice_b": "Soekarno",
            "choice_c": "Habibie",
            "choice_d": "Megawati",
            "answer": "B",
        },
        {
            "question": "Pulau terbesar di Indonesia?",
            "choice_a": "Jawa",
            "choice_b": "Sumatera",
            "choice_c": "Kalimantan",
            "choice_d": "Sulawesi",
            "answer": "C",
        },
    ],
}

runner = CliRunner()


def _dataset_side_effect(
    self: object, split: str, limit: int | None = None
) -> list[dict[str, object]]:
    from nusabench.tasks.base import Task

    assert isinstance(self, Task)
    task_name: str = self.config.task  # type: ignore[union-attr]
    data = TASK_SAMPLE_DATASETS.get(task_name, TASK_SAMPLE_DATASETS["sentiment_smsa"])
    return data[: limit or len(data)]


def _stub_compute_metrics(
    self: object,
    task: object,
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    return {"stub_metric": 0.5}


def test_all_task_configs_load() -> None:
    yaml_files = sorted(CONFIGS_DIR.glob("*.yaml"))
    assert len(yaml_files) == 8, f"Expected 8 YAML configs, found {len(yaml_files)}"
    for yaml_path in yaml_files:
        config = load_task_config(yaml_path)
        assert config.task, f"Empty task name in {yaml_path.name}"
        assert config.task in ALL_TASK_NAMES, f"Unexpected task {config.task}"


def test_dummy_model_runs_all_tasks() -> None:
    with (
        patch(
            "nusabench.tasks.base.Task.load_dataset",
            side_effect=_dataset_side_effect,
            autospec=True,
        ),
        patch(
            "nusabench.evaluator.Evaluator._compute_metrics",
            side_effect=_stub_compute_metrics,
            autospec=True,
        ),
    ):
        result = nb.evaluate(model="dummy", tasks=ALL_TASK_NAMES, limit=3)

    assert isinstance(result, EvaluationResult)
    assert result.model == "dummy"
    for task_name in ALL_TASK_NAMES:
        assert task_name in result.results, f"Missing result for {task_name}"
        assert result.results[task_name].num_samples > 0


def test_cli_evaluate_dummy_single_task(tmp_path: Path) -> None:
    output_file = str(tmp_path / "nb_test.json")

    with (
        patch(
            "nusabench.tasks.base.Task.load_dataset",
            side_effect=_dataset_side_effect,
            autospec=True,
        ),
        patch(
            "nusabench.evaluator.Evaluator._compute_metrics",
            side_effect=_stub_compute_metrics,
            autospec=True,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy",
                "--task",
                "sentiment_smsa",
                "--limit",
                "2",
                "--output",
                output_file,
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    data = json.loads(Path(output_file).read_text(encoding="utf-8"))
    assert "sentiment_smsa" in data["results"]


def test_cli_list_tasks_shows_all_eight() -> None:
    result = runner.invoke(app, ["list-tasks"])
    assert result.exit_code == 0
    for task_name in ALL_TASK_NAMES:
        assert task_name in result.output, f"Missing {task_name} in list-tasks output"


def test_cli_list_models_shows_backends() -> None:
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    for backend in ("dummy", "hf", "gemini"):
        assert backend in result.output, f"Missing {backend} in list-models output"


def test_cli_evaluate_multi_task(tmp_path: Path) -> None:
    output_file = str(tmp_path / "multi.json")
    multi_tasks = ["sentiment_smsa", "nli_wrete", "qa_facqa"]

    with (
        patch(
            "nusabench.tasks.base.Task.load_dataset",
            side_effect=_dataset_side_effect,
            autospec=True,
        ),
        patch(
            "nusabench.evaluator.Evaluator._compute_metrics",
            side_effect=_stub_compute_metrics,
            autospec=True,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy",
                "--task",
                "sentiment_smsa",
                "--task",
                "nli_wrete",
                "--task",
                "qa_facqa",
                "--limit",
                "2",
                "--output",
                output_file,
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    data = json.loads(Path(output_file).read_text(encoding="utf-8"))
    for task_name in multi_tasks:
        assert task_name in data["results"], f"Missing {task_name} in JSON output"


def test_cli_invalid_task_error() -> None:
    result = runner.invoke(
        app,
        ["evaluate", "--model", "dummy", "--task", "nonexistent_xyz", "--limit", "1"],
    )
    assert result.exit_code != 0 or "error" in result.output.lower()


def test_cli_invalid_model_error() -> None:
    result = runner.invoke(
        app,
        ["evaluate", "--model", "badmodel_xyz", "--task", "sentiment_smsa", "--limit", "1"],
    )
    assert result.exit_code != 0 or "error" in result.output.lower()


def test_json_output_structure(tmp_path: Path) -> None:
    output_file = str(tmp_path / "structure.json")

    with (
        patch(
            "nusabench.tasks.base.Task.load_dataset",
            side_effect=_dataset_side_effect,
            autospec=True,
        ),
        patch(
            "nusabench.evaluator.Evaluator._compute_metrics",
            side_effect=_stub_compute_metrics,
            autospec=True,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy",
                "--task",
                "sentiment_smsa",
                "--limit",
                "2",
                "--output",
                output_file,
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    data = json.loads(Path(output_file).read_text(encoding="utf-8"))
    assert "model" in data
    assert data["model"] == "dummy"
    assert "results" in data
    assert "nusabench_version" in data
    assert "timestamp" in data
    assert isinstance(data["results"], dict)
