from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest
from typing_extensions import override

from nusabench import EvaluationResult, TaskResult, evaluate
from nusabench.evaluator import Evaluator
from nusabench.models import DummyModel, Model
from nusabench.tasks import Task, TaskRegistry
from nusabench.tasks.base import TaskConfig
from nusabench.tasks.loader import load_task_config
from nusabench.utils.config import NusaBenchConfig

# pyright: reportPrivateUsage=false

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DATASET: list[dict[str, object]] = [
    {"text": "contoh 1", "label": "positif"},
    {"text": "contoh 2", "label": "positif"},
    {"text": "contoh 3", "label": "positif"},
]


def build_task_config(
    *,
    task: str = "test_task",
    output_type: str = "generate_until",
    target_choices: list[str] | None = None,
    generation_kwargs: dict[str, object] | None = None,
    metric_list: list[dict[str, object]] | None = None,
) -> TaskConfig:
    return TaskConfig(
        task=task,
        dataset_path="dummy/dataset",
        dataset_name=None,
        output_type=output_type,
        train_split=None,
        validation_split=None,
        test_split="test",
        doc_to_text="Teks: {{text}}",
        doc_to_target="{{label}}",
        metric_list=metric_list or [{"metric": "exact_match", "aggregation": "mean"}],
        target_choices=target_choices,
        generation_kwargs=generation_kwargs or {},
    )


class ExposedEvaluator(Evaluator):
    def run_multiple_choice(self, task: Task, prompts: list[str]) -> list[str]:
        return self._run_multiple_choice(task, prompts)

    def run_task(self, task: Task, dataset: list[dict[str, object]]) -> list[str]:
        return self._run_task(task, dataset)


def test_basic_evaluation_returns_expected_structure() -> None:
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))
    evaluator = Evaluator(model=DummyModel(), tasks=[task], config=NusaBenchConfig(limit=3))

    with patch.object(task, "load_dataset", return_value=SAMPLE_DATASET):
        result = evaluator.evaluate()

    assert isinstance(result, EvaluationResult)
    assert result.model == "dummy"
    assert "dummy_task" in result.results
    task_result = result.results["dummy_task"]
    assert isinstance(task_result, TaskResult)
    assert task_result.num_samples == 3
    assert task_result.model_name == "dummy"
    assert isinstance(task_result.metrics, dict)
    assert task_result.task_name == "dummy_task"


def test_evaluate_convenience_function_uses_registered_task() -> None:
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))
    TaskRegistry.register(task)

    with patch("nusabench.tasks.base.Task.load_dataset", return_value=SAMPLE_DATASET[:2]):
        result = evaluate(model="dummy", tasks=["dummy_task"], limit=2)

    assert result.model == "dummy"
    assert result.results["dummy_task"].num_samples == 2


def test_evaluator_passes_limit_to_load_dataset() -> None:
    task = Task(load_task_config(FIXTURES_DIR / "dummy_task.yaml"))
    evaluator = Evaluator(model=DummyModel(), tasks=[task], config=NusaBenchConfig(limit=2))

    with patch.object(task, "load_dataset", return_value=SAMPLE_DATASET[:2]) as load_dataset:
        _ = evaluator.evaluate()

    load_dataset.assert_called_once_with(split="test", limit=2)


def test_run_multiple_choice_with_loglikelihood_model_returns_choices() -> None:
    task = Task(
        build_task_config(
            task="mc_task",
            output_type="multiple_choice",
            target_choices=["positif", "negatif", "netral"],
        )
    )
    evaluator = ExposedEvaluator(model=DummyModel(), tasks=[task])

    predictions = evaluator.run_multiple_choice(task, ["prompt 1", "prompt 2"])

    assert predictions == ["positif", "positif"]
    assert all(prediction in ["positif", "negatif", "netral"] for prediction in predictions)


class GenerateOnlyModel(Model):
    def __init__(self) -> None:
        self.generate_mock: Mock = Mock(return_value=["A", "A"])

    @override
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        **kwargs: object,
    ) -> list[str]:
        return cast(list[str], self.generate_mock(prompts, max_tokens=max_tokens, **kwargs))

    @property
    @override
    def model_name(self) -> str:
        return "generate-only"

    @override
    def supports_loglikelihood(self) -> bool:
        return False


def test_run_multiple_choice_with_generate_only_model_maps_letter_to_choice() -> None:
    task = Task(
        build_task_config(
            task="mc_generate_task",
            output_type="multiple_choice",
            target_choices=["positif", "negatif", "netral"],
        )
    )
    model = GenerateOnlyModel()
    evaluator = ExposedEvaluator(model=model, tasks=[task])

    predictions = evaluator.run_multiple_choice(task, ["prompt 1", "prompt 2"])

    assert predictions == ["positif", "positif"]


def test_run_multiple_choice_raises_when_target_choices_missing() -> None:
    task = Task(build_task_config(task="mc_missing_choices", output_type="multiple_choice"))
    evaluator = ExposedEvaluator(model=DummyModel(), tasks=[task])

    with pytest.raises(ValueError, match="target_choices"):
        _ = evaluator.run_multiple_choice(task, ["prompt"])


def test_evaluate_raises_for_invalid_task_name() -> None:
    with pytest.raises(ValueError, match="Unknown task 'nonexistent'"):
        _ = evaluate(model="dummy", tasks=["nonexistent"])


def test_evaluate_raises_for_invalid_model_name() -> None:
    with pytest.raises(ValueError, match="Unknown model 'nonexistent'"):
        _ = evaluate(model="nonexistent", tasks=["dummy_task"])


def test_run_task_generate_until_uses_model_generate() -> None:
    task = Task(build_task_config(task="generate_task", output_type="generate_until"))
    model = GenerateOnlyModel()
    evaluator = ExposedEvaluator(model=model, tasks=[task])

    predictions = evaluator.run_task(task, SAMPLE_DATASET[:2])

    assert predictions == ["A", "A"]
    model.generate_mock.assert_called_once()


def test_run_task_forwards_generation_kwargs() -> None:
    task = Task(
        build_task_config(
            task="generate_with_kwargs",
            output_type="generate_until",
            generation_kwargs={"max_tokens": 50},
        )
    )
    model = GenerateOnlyModel()
    evaluator = ExposedEvaluator(model=model, tasks=[task])

    _ = evaluator.run_task(task, SAMPLE_DATASET[:2])

    model.generate_mock.assert_called_once_with(
        ["Teks: contoh 1", "Teks: contoh 2"],
        max_tokens=50,
    )
